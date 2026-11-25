# airllm-lab

Exhaustive AirLLM lab: benchmarks → framework → lab on macOS with MLX.

## Prerequisites

- `uv` 0.13.8 (`curl -LsSf astral.sh/uv/install.sh | sh` for a user-level
  install; never `sudo`). The plan never installs `uv` itself.
- Python: `UV_PYTHON=3.14` pinned via `uv python pin 3.14` (Step 1.1 first
  probed `UV_PYTHON=3.13`; Step 1.3R re-pinned to 3.14 after verifying `cp314`
  macOS-arm64 wheels; this tree runs Python 3.14.8 in `.venv`).
- Disk: keep ~2x the model size free (AirLLM splits the checkpoint into layer
  shards beside the cache; `df -h /` showed 494Gi avail at plan time).
- macOS limit: the macOS MLX backend runs **Llama-style models only**
  (`LlamaForCausalLM`, `MistralForCausalLM`). Anything else raises before any
  download:
  `ValueError: Qwen2ForCausalLM is not supported on macOS yet. The macOS backend (MLX) only runs Llama-style models: LlamaForCausalLM, MistralForCausalLM.`
  (source: `air_llm/airllm/auto_model.py:65-74` at pinned SHA
  `291a93ab45effeb5c2c0353ca0be2f690af7a616`, tag `v4.0.1`).
- Training is CUDA-gated upstream (`train_qwen38_lora.py:50-52`):
  `CUDA is required for this trainer.` — host-blocked on this Mac, recorded
  verbatim, never faked.

## Install

```bash
uv sync --frozen        # creates local .venv (uv-managed; never pip install / sudo)
```

Notebook tooling (`jupytext>=1.16`, `papermill>=2.6`, `ipykernel>=6.29`) and the
macOS backend dep (`mlx==0.32.4` — darwin `import airllm` fails without it)
are repo dependencies, so they come from the same `.venv`. Every run uses
`uv run --frozen -- ...`.

Register the venv kernel once (user-level, kept after the plan per the
Retained table):

```bash
uv run --frozen -- python -m ipykernel install --user --name airllm-lab \
  --display-name "airllm-lab (.venv)"
```

## Runnable examples

Every script example is mirrored by a notebook. Minimal first, full-matrix last.

| # | Script (`uv run --frozen -- ...`) | Notebook mirror |
|---|---|---|
| 1 | Minimal generate: `python -c "from airllm_lab.load import load_model; from airllm_lab.generate import generate_text; m = load_model('TinyLlama/TinyLlama-1.1B-Chat-v1.0'); print(generate_text(m, ['Hello world']))"` | `notebooks/inference/02-minimal-generate.ipynb` |
| 2 | Inference surface (chat + profiling/prefetch matrix): `python bench/run_bench.py` (PRIMARY 2x2 + chat/raw) | `notebooks/inference/03-chat-profiling.ipynb` |
| 3 | Training surface `--help` + `--steps` smoke: `python -m airllm_lab.train --help` (module passthrough: `sys.exit(train_surface())`) | `notebooks/training/04-train-surface.ipynb` |
| 4 | Full matrix on PRIMARY/ALTERNATE/NEGATIVE: `python bench/run_bench.py` → `bench/results.csv` | `notebooks/benchmarking/05-tiny-matrix.ipynb` |
| 0 | Import + tiny 8-token smoke (start here) | `notebooks/testing/01-smoke.ipynb` |

Pinned IDs (Step 1.2 inventory, ungated, safetensors-confirmed):

- PRIMARY `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (~1.1B, Llama-arch, chat template)
- ALTERNATE `HuggingFaceTB/SmolLM-135M-Instruct` (~135M, Llama-arch, chat template)
- NEGATIVE `Qwen/Qwen2-0.5B` (~0.5B, reserved for the macOS negative path)

## Notebooks

Tree map (run order 01 → 05):

```text
notebooks/
  README.md                          map + run order + params (this section's source)
  testing/01-smoke.ipynb             venv import check + PRIMARY 8-token generate
  inference/02-minimal-generate.ipynb
  inference/03-chat-profiling.ipynb
  training/04-train-surface.ipynb
  benchmarking/05-tiny-matrix.ipynb
  executed/                          papermill outputs (ignored, never committed)
```

Each `.ipynb` is paired with a `.py` (percent format) via `[tool.jupytext]`
(`formats = "notebooks//ipynb,notebooks//py:percent"`). Workflow after editing
either side:

```bash
uv run --frozen -- jupytext --sync notebooks/inference/02-minimal-generate.ipynb
uv run --frozen -- jupytext --check notebooks/testing/01-smoke.ipynb \
  notebooks/inference/02-minimal-generate.ipynb \
  notebooks/inference/03-chat-profiling.ipynb \
  notebooks/training/04-train-surface.ipynb \
  notebooks/benchmarking/05-tiny-matrix.ipynb
```

Execute all five with tiny params into `notebooks/executed/` (ignored):

```bash
uv run --frozen -- papermill notebooks/testing/01-smoke.ipynb notebooks/executed/01-smoke.ipynb \
  -k airllm-lab -p MAX_NEW_TOKENS 8 -p MODEL_ID TinyLlama/TinyLlama-1.1B-Chat-v1.0
uv run --frozen -- papermill notebooks/inference/02-minimal-generate.ipynb notebooks/executed/02-minimal-generate.ipynb \
  -k airllm-lab -p MAX_NEW_TOKENS 8 -p MODEL_ID TinyLlama/TinyLlama-1.1B-Chat-v1.0
uv run --frozen -- papermill notebooks/inference/03-chat-profiling.ipynb notebooks/executed/03-chat-profiling.ipynb \
  -k airllm-lab -p MAX_NEW_TOKENS 8 -p MODEL_ID TinyLlama/TinyLlama-1.1B-Chat-v1.0
uv run --frozen -- papermill notebooks/training/04-train-surface.ipynb notebooks/executed/04-train-surface.ipynb \
  -k airllm-lab -p MAX_NEW_TOKENS 8 -p MODEL_ID TinyLlama/TinyLlama-1.1B-Chat-v1.0
uv run --frozen -- papermill notebooks/benchmarking/05-tiny-matrix.ipynb notebooks/executed/05-tiny-matrix.ipynb \
  -k airllm-lab -p MAX_NEW_TOKENS 8 -p MODEL_ID TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

The training notebook records smoke success or a verbatim BLOCKED cell when the
host blocks training — never a fake success.

## Options (7 knobs, verified at v4.0.1 / SHA 291a93ab)

| # | Knob | Generic default (`airllm_base.py:84-87`) | macOS MLX default (`airllm_llama_mlx.py:233-236`) | Status |
|---|---|---|---|---|
| 1 | `device` | `None` → `default_device()` (cuda/mps/cpu) | `"cuda:0"` (accepted, never used in body) | verified |
| 2 | `layer_shards_saving_path` | `None` (next to model cache) | `None` | verified |
| 3 | `hf_token` | `None` | `None` | verified |
| 4 | `delete_original` | `False` | `False` | verified |
| 5 | `prefetching` | `True` (force-disabled when `compression` set) | `True` | verified |
| 6 | `compression` | `None` (`'4bit'`/`'8bit'` need bitsandbytes) | `None` | verified |
| 7 | `profiling_mode` | `False` | `False` | verified |

Zero knobs UNVERIFIED — all seven read verbatim from source at the pinned SHA.
The wrapper default is `device="cpu"` (explicit CPU keeps a CPU-only host off
the incidental `cuda:0` path).

## References

- Upstream source: `https://github.com/lyogavin/airllm`
