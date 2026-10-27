# %% Cell 1: install packages
# Turn Internet on in the notebook settings first. Run this cell, then restart the session if Kaggle asks.
import subprocess, sys
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "bitsandbytes"],
    check=True,
)
print("Packages installed.")

# %% Cell 2: settings and environment record
import gc, json, os, statistics, time
from datetime import datetime, timezone

os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # use one T4 only, so results are clean

import torch
import transformers
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

assert torch.cuda.is_available(), "No GPU found. Turn on the accelerator in the notebook settings."
DEVICE = "cuda:0"

# Check this figure against NVIDIA's T4 datasheet before you publish anything.
T4_BANDWIDTH_GBPS = 320

PROMPT_TOKENS = 512
NEW_TOKENS = 256
BATCH_SIZES = [1, 4, 8]
RUNS = 3

# To skip the 7B model (about 15 GB download), delete its line below.
MODELS = [
    {"label": "Qwen2.5 3B Instruct, FP16", "id": "Qwen/Qwen2.5-3B-Instruct", "quant": None},
    {"label": "Qwen2.5 7B Instruct, 4 bit", "id": "Qwen/Qwen2.5-7B-Instruct", "quant": "4bit"},
]

gpu_query = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
    capture_output=True, text=True,
).stdout.strip()

environment = {
    "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "gpu": gpu_query,
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "prompt_tokens": PROMPT_TOKENS,
    "new_tokens": NEW_TOKENS,
    "batch_sizes": BATCH_SIZES,
    "runs_per_setting": RUNS,
    "t4_bandwidth_gbps_assumed": T4_BANDWIDTH_GBPS,
}
print(json.dumps(environment, indent=2))

# %% Cell 3: measurement functions
def build_batch(tokenizer, batch_size):
    # A fixed prompt cut to exactly PROMPT_TOKENS tokens, repeated across the batch.
    text = "The quick brown fox jumps over the lazy dog. " * 200
    ids = tokenizer(text, return_tensors="pt").input_ids[0][:PROMPT_TOKENS]
    assert len(ids) == PROMPT_TOKENS, "Prompt is shorter than expected."
    input_ids = ids.unsqueeze(0).repeat(batch_size, 1).to(DEVICE)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def timed_generate(model, inputs, new_tokens, pad_id):
    # Greedy decoding with a forced length, so every run produces the same number of tokens.
    torch.cuda.synchronize()
    start = time.perf_counter()
    model.generate(
        **inputs,
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        do_sample=False,
        repetition_penalty=1.0,
        pad_token_id=pad_id,
    )
    torch.cuda.synchronize()
    return time.perf_counter() - start


def run_model(cfg):
    rows = []
    tokenizer = AutoTokenizer.from_pretrained(cfg["id"])
    kwargs = {"device_map": {"": 0}}
    if cfg["quant"] == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(cfg["id"], **kwargs)
    model.eval()
    footprint_gb = model.get_memory_footprint() / 1e9
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    print(f"{cfg['label']}: weights use {footprint_gb:.2f} GB")

    for batch in BATCH_SIZES:
        inputs = build_batch(tokenizer, batch)
        batch_rows = []
        try:
            with torch.no_grad():
                timed_generate(model, inputs, 8, pad_id)  # warm up, not recorded
                torch.cuda.reset_peak_memory_stats()
                for run in range(1, RUNS + 1):
                    t_first = timed_generate(model, inputs, 1, pad_id)
                    t_full = timed_generate(model, inputs, NEW_TOKENS, pad_id)
                    decode_time = max(t_full - t_first, 1e-9)
                    per_stream = (NEW_TOKENS - 1) / decode_time
                    batch_rows.append({
                        "model": cfg["label"],
                        "batch": batch,
                        "run": run,
                        "ttft_s": round(t_first, 3),
                        "per_stream_tok_s": round(per_stream, 2),
                        "aggregate_tok_s": round(per_stream * batch, 2),
                        "footprint_gb": round(footprint_gb, 3),
                        "note": "",
                    })
                    print(f"  batch {batch} run {run}: ttft {t_first:.2f}s, {per_stream * batch:.1f} tok/s total")
            peak = torch.cuda.max_memory_allocated() / 1e9
            for r in batch_rows:
                r["peak_gb"] = round(peak, 2)
            rows.extend(batch_rows)
        except torch.cuda.OutOfMemoryError:
            print(f"  batch {batch}: out of memory, skipping larger batches")
            rows.append({"model": cfg["label"], "batch": batch, "run": 0, "note": "out of memory"})
            torch.cuda.empty_cache()
            break

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows

# %% Cell 4: run everything (this takes a while, mostly downloads)
all_rows = []
for cfg in MODELS:
    all_rows.extend(run_model(cfg))
print("Done.")

# %% Cell 5: summary, roofline comparison and saved files
df = pd.DataFrame(all_rows)
ok = df[df["note"] == ""]

summary = ok.groupby(["model", "batch"]).agg(
    ttft_s_median=("ttft_s", "median"),
    tok_s_median=("aggregate_tok_s", "median"),
    tok_s_min=("aggregate_tok_s", "min"),
    tok_s_max=("aggregate_tok_s", "max"),
    peak_gb=("peak_gb", "max"),
    footprint_gb=("footprint_gb", "first"),
).reset_index()

# Roofline estimate: each decoded token reads all the weights once, so the speed limit is
# memory bandwidth divided by weight size. A batch reads the weights once for all its streams.
# This ignores the KV cache and compute limits, so treat it as an upper bound.
summary["roofline_tok_s"] = T4_BANDWIDTH_GBPS / summary["footprint_gb"] * summary["batch"]
summary["share_of_roofline"] = summary["tok_s_median"] / summary["roofline_tok_s"]
summary = summary.round(3)

out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
df.to_csv(os.path.join(out_dir, "inference_benchmark_runs.csv"), index=False)
summary.to_csv(os.path.join(out_dir, "inference_benchmark_summary.csv"), index=False)
with open(os.path.join(out_dir, "environment.json"), "w") as f:
    json.dump(environment, f, indent=2)

print(summary.to_string(index=False))
print("\nSaved inference_benchmark_runs.csv, inference_benchmark_summary.csv and environment.json to", out_dir)
