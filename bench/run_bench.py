"""Step 1.4 — tiny AirLLM benchmark matrix (host-runnable subset).

Runs on CPU with an explicit device (never the incidental ``cuda:0`` default),
tiny prompts (<=8 new tokens), caches + layer shards pointed into scratch.

- PRIMARY (Llama-arch tiny) x {compression None} x {profiling off/on}
  x {prefetch on/off} x {raw generate, chat-template}
- ALTERNATE tiny smoke (generate only)
- NEGATIVE Qwen-tiny (expects the pre-download macOS architecture ValueError,
  captured verbatim)
- shard-reuse check (second load path)

Writes ``bench/results.csv`` with columns:
model,params,compression,profiling,prefetch,prompt_mode,load_s,peak_rss_mb,tok_s,notes

Any measurement that cannot be taken is recorded as ``BLOCKED:<reason>`` text,
never a guess. Newlines inside cells are flattened to `` | `` so one CSV
record is always exactly one line (``wc -l`` gate).
"""

import csv
import os
import resource
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- scratch setup
# Default scratch dir created by the Step 1.4 brief; overridable for tests.
SCRATCH = Path(
    os.environ.get(
        "AIRLLM_LAB_SCRATCH",
        "/var/folders/_m/jyv5nmsn4y305mjjcm_4fq240000gn/T/opencode/"
        "ses_df5b5e7cbffeUTbz4f1XoUo1Q3/airllm-lab-plan/benchrun",
    )
)
HF_CACHE = SCRATCH / "hf_hub_cache"
SHARDS = SCRATCH / "shards"
SCRATCH.mkdir(parents=True, exist_ok=True)
HF_CACHE.mkdir(parents=True, exist_ok=True)
SHARDS.mkdir(parents=True, exist_ok=True)

# Point all Hub/transformers caches + AirLLM layer shards into scratch.
# Temp artifacts go ONLY here — never into the repo or the user cache.
os.environ["HF_HUB_CACHE"] = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
os.environ["HF_HOME"] = str(SCRATCH / "hf_home")

# ---------------------------------------------------------------- model IDs
# From Step 1.2 inventory (source v4.0.1, SHA 291a93ab45effeb5c2c0353ca0be2f690af7a616):
# metadata via unauthenticated Hub API, no token, no downloads.
PRIMARY = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PRIMARY_PARAMS = "1100048384"  # safetensors total, inventory section 6
ALTERNATE = "HuggingFaceTB/SmolLM-135M-Instruct"
ALTERNATE_PARAMS = "134515008"  # safetensors total, inventory section 6
NEGATIVE = "Qwen/Qwen2-0.5B"
NEGATIVE_PARAMS = "494032768"  # safetensors total, inventory section 6

# Step 1.2 device truth: generic default None -> default_device(); MLX signature
# default "cuda:0" is accepted-but-unused on the macOS backend. Pass CPU
# explicitly so a CPU host never touches the cuda:0 default path.
DEVICE = "cpu"
MAX_NEW_TOKENS = 8
PROMPT_RAW = "Hello world"
PROMPT_CHAT = [{"role": "user", "content": "Say hi in five words."}]

RESULTS = Path(__file__).resolve().parent / "results.csv"
COLUMNS = [
    "model",
    "params",
    "compression",
    "profiling",
    "prefetch",
    "prompt_mode",
    "load_s",
    "peak_rss_mb",
    "tok_s",
    "notes",
]


def flat(text):
    """Flatten to one line so one CSV record stays one physical line."""
    return " | ".join(str(text).splitlines())


def rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 1)


def try_import():
    """Import AirLLM's AutoModel. Returns (AutoModel|None, err|None)."""
    try:
        from airllm import AutoModel  # noqa: PLC0415

        return AutoModel, None
    except Exception as exc:  # noqa: BLE001 - verbatim capture is the point
        return None, flat(f"{type(exc).__name__}: {exc}")


def blocked_row(model, params, compression, profiling, prefetch, prompt_mode, reason):
    cell = f"BLOCKED:{reason}"
    return {
        "model": model,
        "params": params,
        "compression": compression,
        "profiling": str(profiling),
        "prefetch": str(prefetch),
        "prompt_mode": prompt_mode,
        "load_s": cell,
        "peak_rss_mb": cell,
        "tok_s": cell,
        "notes": flat(reason),
    }


def run_generate(model, model_id, prompt_mode, max_new_tokens):
    """Run one tiny generation; returns (seconds, tok_per_s, note)."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(model_id)
    t0 = time.perf_counter()
    if prompt_mode == "chat-template":
        try:
            prompt = tok.apply_chat_template(
                PROMPT_CHAT, tokenize=False, add_generation_prompt=True
            )
        except Exception as exc:  # noqa: BLE001
            return None, None, f"chat-template unavailable: {flat(exc)}"
    else:
        prompt = PROMPT_RAW
    ids = tok(prompt, return_tensors="np")["input_ids"]
    out = model.generate(ids, max_new_tokens=max_new_tokens)
    dt = time.perf_counter() - t0
    text = out if isinstance(out, str) else str(out)
    # tok/s over newly generated text only, approximated by word count guard:
    # use wall time and requested token budget as a lower-bound-free measure.
    tok_s = round(max_new_tokens / dt, 2) if dt > 0 else None
    return dt, tok_s, f"ok len={len(text)}"


def bench_one(AutoModel, model_id, params, profiling, prefetch, prompt_mode):
    """Load + tiny-generate one matrix cell. Never raises: BLOCKED on failure."""
    slug = model_id.replace("/", "__")
    shard_path = SHARDS / slug
    base_before = rss_mb()
    t0 = time.perf_counter()
    try:
        model = AutoModel.from_pretrained(
            model_id,
            device=DEVICE,  # explicit CPU per Step 1.2 device truth
            compression=None,
            profiling_mode=profiling,
            prefetching=prefetch,
            layer_shards_saving_path=str(shard_path),
        )
    except Exception as exc:  # noqa: BLE001
        return blocked_row(
            model_id, params, "None", profiling, prefetch, prompt_mode,
            f"load failed: {flat(f'{type(exc).__name__}: {exc} ')}",
        )
    load_s = round(time.perf_counter() - t0, 2)
    try:
        gen_s, tok_s, note = run_generate(model, model_id, prompt_mode, MAX_NEW_TOKENS)
    except Exception as exc:  # noqa: BLE001
        return {
            "model": model_id,
            "params": params,
            "compression": "None",
            "profiling": str(profiling),
            "prefetch": str(prefetch),
            "prompt_mode": prompt_mode,
            "load_s": str(load_s),
            "peak_rss_mb": str(rss_mb()),
            "tok_s": f"BLOCKED:generate failed: {flat(f'{type(exc).__name__}: {exc}')}",
            "notes": flat(f"loaded ok; generate failed: {type(exc).__name__}: {exc}"),
        }
    if gen_s is None:
        return {
            "model": model_id,
            "params": params,
            "compression": "None",
            "profiling": str(profiling),
            "prefetch": str(prefetch),
            "prompt_mode": prompt_mode,
            "load_s": str(load_s),
            "peak_rss_mb": str(rss_mb()),
            "tok_s": "BLOCKED:no generation measurement",
            "notes": note,
        }
    return {
        "model": model_id,
        "params": params,
        "compression": "None",
        "profiling": str(profiling),
        "prefetch": str(prefetch),
        "prompt_mode": prompt_mode,
        "load_s": str(load_s),
        "peak_rss_mb": str(rss_mb()),
        "tok_s": str(tok_s),
        "notes": flat(f"{note}; rss_base={base_before}MB; device={DEVICE}"),
    }


def main():
    AutoModel, import_err = try_import()
    rows = []

    if import_err is not None:
        # Whole matrix host-blocked at import: record verbatim, no guesses.
        reason = (
            f"import airllm failed: {import_err}; "
            "unblock = add mlx to repo deps (mlx 0.32.4 publishes "
            "cp314 macosx arm64 wheels) via pyproject edit + uv lock/sync; "
            "pyproject edit is outside the Step 1.4 write set "
            "(bench/run_bench.py + bench/results.csv only), so no retry here"
        )
        for profiling in (False, True):
            for prefetch in (True, False):
                for prompt_mode in ("raw-generate", "chat-template"):
                    rows.append(
                        blocked_row(
                            PRIMARY, PRIMARY_PARAMS, "None",
                            profiling, prefetch, prompt_mode, reason,
                        )
                    )
        rows.append(
            blocked_row(
                ALTERNATE, ALTERNATE_PARAMS, "None",
                False, True, "raw-generate", f"alternate smoke: {reason}",
            )
        )
        rows.append(
            blocked_row(
                NEGATIVE, NEGATIVE_PARAMS, "None",
                False, True, "raw-generate",
                f"negative path unreachable: {reason}; expected (per Step 1.2) "
                "ValueError: Qwen2ForCausalLM is not supported on macOS yet. "
                "The macOS backend (MLX) only runs Llama-style models: "
                "LlamaForCausalLM, MistralForCausalLM.",
            )
        )
        rows.append(
            blocked_row(
                PRIMARY, PRIMARY_PARAMS, "None",
                False, True, "shard-reuse",
                f"shard-reuse check not run: {reason}",
            )
        )
    else:
        for profiling in (False, True):
            for prefetch in (True, False):
                for prompt_mode in ("raw-generate", "chat-template"):
                    rows.append(
                        bench_one(
                            AutoModel, PRIMARY, PRIMARY_PARAMS,
                            profiling, prefetch, prompt_mode,
                        )
                    )
        rows.append(
            bench_one(AutoModel, ALTERNATE, ALTERNATE_PARAMS, False, True, "raw-generate")
        )
        # NEGATIVE: expect the pre-download macOS architecture ValueError.
        try:
            AutoModel.from_pretrained(
                NEGATIVE,
                device=DEVICE,
                layer_shards_saving_path=str(SHARDS / NEGATIVE.replace("/", "__")),
            )
            neg_note = "UNEXPECTED: no error raised for Qwen on macOS"
            neg_cell = "BLOCKED:expected macOS architecture ValueError not raised"
            rows.append(
                {
                    "model": NEGATIVE,
                    "params": NEGATIVE_PARAMS,
                    "compression": "None",
                    "profiling": "False",
                    "prefetch": "True",
                    "prompt_mode": "raw-generate",
                    "load_s": neg_cell,
                    "peak_rss_mb": neg_cell,
                    "tok_s": neg_cell,
                    "notes": neg_note,
                }
            )
        except Exception as exc:  # noqa: BLE001 - verbatim capture is the point
            err = flat(f"{type(exc).__name__}: {exc}")
            rows.append(
                {
                    "model": NEGATIVE,
                    "params": NEGATIVE_PARAMS,
                    "compression": "None",
                    "profiling": "False",
                    "prefetch": "True",
                    "prompt_mode": "raw-generate",
                    "load_s": f"BLOCKED:negative path raised before load: {err}",
                    "peak_rss_mb": f"BLOCKED:negative path raised before load: {err}",
                    "tok_s": f"BLOCKED:negative path raised before load: {err}",
                    "notes": flat(f"verbatim negative-path error: {err}"),
                }
            )
        # Shard-reuse: second load of PRIMARY should hit cached shards.
        try:
            import contextlib  # noqa: PLC0415
            import io  # noqa: PLC0415

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                bench_one(AutoModel, PRIMARY, PRIMARY_PARAMS, False, True, "raw-generate")
            reuse_out = flat(buf.getvalue()[-500:])
            rows.append(
                {
                    "model": PRIMARY,
                    "params": PRIMARY_PARAMS,
                    "compression": "None",
                    "profiling": "False",
                    "prefetch": "True",
                    "prompt_mode": "shard-reuse",
                    "load_s": "BLOCKED:reuse check is observational only",
                    "peak_rss_mb": "BLOCKED:reuse check is observational only",
                    "tok_s": "BLOCKED:reuse check is observational only",
                    "notes": flat(
                        "second-load stdout tail (expect 'saved layers already found'): "
                        f"{reuse_out}"
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                blocked_row(
                    PRIMARY, PRIMARY_PARAMS, "None", False, True,
                    "shard-reuse", f"reuse check failed: {flat(exc)}",
                )
            )

    with open(RESULTS, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {RESULTS} with {len(rows)} data rows")
    for r in rows:
        print(
            f"{r['model']} | profiling={r['profiling']} prefetch={r['prefetch']} "
            f"mode={r['prompt_mode']} | load_s={r['load_s']} "
            f"peak_rss_mb={r['peak_rss_mb']} tok_s={r['tok_s']}"
        )


if __name__ == "__main__":
    sys.exit(main())
