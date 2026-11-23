# airllm-lab

Placeholder scope: exhaustive AirLLM lab (benchmarks → framework → lab) on this
macOS host, per plan
`/Users/mavasbel/Dev/repos/docs/plans/2026-11-23-15-27-airllm-exhaustive-lab-plan.md`.

- Toolchain: `uv`, `UV_PYTHON=3.13` (Step 1.1 decision).
- AirLLM pinned: `4.0.1` (Step 1.1). Source truth: inventory at Step 1.2
  (`v4.0.1`, SHA `291a93ab45effeb5c2c0353ca0be2f690af7a616`).
- Install: `uv sync --frozen` inside this repo (produces local `.venv`).
- Every run: `uv run --frozen -- ...`.
- Notebook content (`.ipynb` files) arrives in Step 3.1; this skeleton only
  carries the paired directory tree.
