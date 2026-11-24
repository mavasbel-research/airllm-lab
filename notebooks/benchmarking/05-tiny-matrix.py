# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.7
#   kernelspec:
#     display_name: airllm-lab (.venv)
#     language: python
#     name: airllm-lab
# ---

# %% [markdown]
# # 05 — Tiny benchmark matrix (`nb-benchmarking-matrix`)
#
# Renders `bench/results.csv` (written by `bench/run_bench.py`) plus the
# blocked-path table. This notebook performs no loads itself; it reports
# measured rows verbatim. Mirrors the full-matrix script example:
# `uv run --frozen -- python bench/run_bench.py`.

# %% tags=["parameters"]
# Parameters kept for uniform papermill invocation (unused by this notebook).
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 8


# %%
import csv
from pathlib import Path

CSV = Path('/Users/mavasbel/Dev/repos/research/airllm-lab/bench/results.csv')
assert CSV.exists(), f'missing {CSV} — run bench/run_bench.py first'
with open(CSV, newline='') as fh:
    rows = list(csv.DictReader(fh))
print(f'{len(rows)} data rows')
print('| model | profiling | prefetch | mode | load_s | peak_rss_mb | tok_s |')
print('|---|---|---|---|---|---|---|')
for r in rows:
    short = r['model'].split('/')[-1]
    print(f"| {short} | {r['profiling']} | {r['prefetch']} | {r['prompt_mode']} | "
          f"{r['load_s']} | {r['peak_rss_mb']} | {r['tok_s']} |")


# %%
blocked = [r for r in rows
           if any(str(v).startswith('BLOCKED') for v in r.values())]
print(f'{len(blocked)} blocked-path rows')
print('| model | mode | reason |')
print('|---|---|---|')
for r in blocked:
    short = r['model'].split('/')[-1]
    print(f"| {short} | {r['prompt_mode']} | {r['notes'][:220]} |")

