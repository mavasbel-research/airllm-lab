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
# # 01 — Smoke test (`nb-testing-smoke`)
#
# Venv import check plus one tiny generate on PRIMARY (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) with `MAX_NEW_TOKENS=8`.
# Run order: first. Tiny params only.

# %% tags=["parameters"]
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 8


# %%
import sys
print(sys.version)
print(sys.prefix)
import airllm_lab.load, airllm_lab.generate, airllm_lab.train
print('airllm_lab imports ok')
import airllm
print('airllm import ok:', airllm.__file__)


# %%
from airllm_lab.load import load_model
from airllm_lab.generate import generate_text

model = load_model(MODEL_ID)
texts = generate_text(model, ['Hello world'], max_new_tokens=MAX_NEW_TOKENS)
print(texts[0])
assert isinstance(texts[0], str) and len(texts[0]) > 0, 'empty completion'
print('smoke ok')

