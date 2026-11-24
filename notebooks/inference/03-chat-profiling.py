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
# # 03 — Chat template + profiling/prefetch (`nb-inference-chat-profiling`)
#
# Chat-template generation on PRIMARY across `profiling_mode` on/off x `prefetching` on/off (4 cells, same matrix as `bench/run_bench.py`).
# Mirrors the inference+training-surface script example for the inference half.

# %% tags=["parameters"]
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 8


# %%
from airllm_lab.load import load_model
from airllm_lab.generate import generate_text

MESSAGES = [{'role': 'user', 'content': 'Say hi in five words.'}]
rows = []
for profiling in (False, True):
    for prefetch in (True, False):
        m = load_model(MODEL_ID, profiling_mode=profiling, prefetching=prefetch)
        texts = generate_text(m, [MESSAGES], max_new_tokens=MAX_NEW_TOKENS)
        rows.append((profiling, prefetch, len(texts[0]), texts[0].replace(chr(10), ' | ')))
        print(f'profiling={profiling} prefetch={prefetch} len={len(texts[0])}')
print()
print('| profiling | prefetch | len | text |')
print('|---|---|---|---|')
for profiling, prefetch, n, t in rows:
    print(f'| {profiling} | {prefetch} | {n} | {t} |')
assert len(rows) == 4 and all(r[2] > 0 for r in rows)

