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
# # 02 — Minimal generate (`nb-inference-minimal`)
#
# Raw (non-chat) generation through the `airllm_lab` wrapper on PRIMARY.
# Mirrors the script example:
# `uv run --frozen -- python -c "from airllm_lab.load import load_model; from airllm_lab.generate import generate_text; print(generate_text(load_model(MODEL_ID), ['Hello world']))"`.
#
# Note: on the macOS MLX backend extra HF kwargs such as `use_cache` or
# `return_dict_in_generate` are swallowed by `AirLLMLlamaMlx.generate`
# (`**kwargs` never referenced in the body — Step 1.2 source truth), so the
# wrapper does not pass them there. On the generic backend they flow through
# transformers delegation. The branch below reports which path ran.

# %% tags=["parameters"]
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 8


# %%
from airllm_lab.load import load_model
from airllm_lab.generate import generate_text

model = load_model(MODEL_ID)
print('backend:', type(model).__name__)
texts = generate_text(model, ['Hello world'], max_new_tokens=MAX_NEW_TOKENS)
print(texts[0])
assert isinstance(texts[0], str) and len(texts[0]) > 0


# %%
is_mlx = type(model).__name__ == 'AirLLMLlamaMlx'
if is_mlx:
    print('MLX backend: use_cache/return_dict_in_generate are swallowed upstream; '
          'not passed (documented, not faked).')
else:
    from transformers import AutoTokenizer
    tok = model.tokenizer
    ids = tok('Hello world', return_tensors='pt')['input_ids']
    out = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS,
                         use_cache=True, return_dict_in_generate=False)
    print(tok.decode(out[0], skip_special_tokens=True))

