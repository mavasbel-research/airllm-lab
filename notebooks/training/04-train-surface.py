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
# # 04 — Training surface (`nb-training-surface`)
#
# Exercises `airllm_lab.train.train_surface`: `--help` (always works) plus a
# `--steps` smoke. Upstream training is CUDA-gated
# (`train_qwen38_lora.py:50-52`: `CUDA is required for this trainer.`), so on
# this macOS host the smoke cell records a verbatim BLOCKED result and a
# non-zero return — never a faked success.

# %% tags=["parameters"]
# Parameters kept for uniform papermill invocation (unused by this notebook).
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 8


# %%
from airllm_lab.train import train_surface

rc_help = train_surface(['--help'])
print(f'--help rc={rc_help}')
assert rc_help == 0


# %%
from airllm_lab.train import train_surface

rc = train_surface(['--steps', '1'])
print(f'--steps 1 rc={rc}')
if rc == 1:
    print('BLOCKED (verbatim upstream gate): CUDA is required for this trainer. '
          'No training was run; nothing faked.')
elif rc == 0:
    print('trainer ran (CUDA host).')
else:
    print(f'trainer surface returned {rc} (see stderr for verbatim reason).')
assert rc in (0, 1, 2), f'unexpected rc={rc}'

