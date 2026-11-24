# Notebooks

Paired notebooks (`.ipynb` + `.py` percent, kept in sync via `jupytext --sync`).
Kernel: `airllm-lab` (`.venv`, display name `airllm-lab (.venv)`).

## Map

| Order | Notebook | Use case | What |
|---|---|---|---|
| 01 | `testing/01-smoke.ipynb` | `nb-testing-smoke` | venv import check + PRIMARY tiny 8-token generate |
| 02 | `inference/02-minimal-generate.ipynb` | `nb-inference-minimal` | raw generate through the wrapper |
| 03 | `inference/03-chat-profiling.ipynb` | `nb-inference-chat-profiling` | chat template + `profiling_mode` on/off + `prefetching` on/off |
| 04 | `training/04-train-surface.ipynb` | `nb-training-surface` | CLI `--help` + `--steps` smoke; verbatim BLOCKED when host-blocked |
| 05 | `benchmarking/05-tiny-matrix.ipynb` | `nb-benchmarking-matrix` | renders `bench/results.csv` + blocked-path table |

## Run order and params

Run 01 → 05. Every notebook takes the same papermill params (tiny only):

- `-p MODEL_ID TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `-p MAX_NEW_TOKENS 8`
- `-k airllm-lab`

Outputs go to `notebooks/executed/` (git-ignored, never committed). See the
top-level README for the exact commands.
