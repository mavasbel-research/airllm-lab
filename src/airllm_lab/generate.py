"""Generate helper: raw + chat-template prompts over either AirLLM backend.

Step 1.2 source truth (airllm v4.0.1, SHA 291a93ab45effeb5c2c0353ca0be2f690af7a616):

- MLX backend (``AirLLMLlamaMlx.generate``, ``airllm_llama_mlx.py:290-302``):
  ``generate(x, temperature=0, max_new_tokens=None, **kwargs)``. ``x`` is
  positional and coerced via ``mx.array(x)``, which rejects ``str`` — the
  caller MUST encode prompts to token-id arrays first (live proof: numpy ids
  ``tok(prompt, return_tensors="np")["input_ids"]`` generate real text, plain
  ``str`` raises ``ValueError: Invalid type str``). ``**kwargs`` is swallowed,
  so a ``streamer=`` kwarg is silently IGNORED. Returns a decoded ``str``.
- Generic backend (``AirLLMBaseModel.generate``, ``airllm_base.py:932-933``):
  pure delegation to the underlying transformers model — full HF ``generate``
  signature (``streamer=`` honored). Returns whatever transformers returns.
"""

from __future__ import annotations

#: Class name of the macOS MLX backend. Branching on it is the seam
#: (justification): the two ``generate`` implementations have incompatible
#: contracts (positional ids in / ``str`` out vs. full HF delegation), and
#: AirLLM exposes no common capability flag — the concrete class is the only
#: source-verified discriminator (``airllm/auto_model.py:74`` returns
#: ``AirLLMLlamaMlx`` on darwin).
_MLX_CLASS_NAME = "AirLLMLlamaMlx"


def _tokenizer_for(model):
    """Return the tokenizer bundled with *model*.

    Seam (justification): both backends already bundle one
    (``airllm_base.py:192``, ``airllm_llama_mlx.py:265``:
    ``self.tokenizer = self.get_tokenizer(...)``), so the wrapper reuses it
    instead of re-downloading a second copy via ``AutoTokenizer``.
    """
    return model.tokenizer


def _ensure_pad_token(tokenizer):
    """Apply the README FAQ pad-token branch: fall back to ``eos_token``.

    Tokenizers of Llama-style chat checkpoints commonly ship without a pad
    token; any padded/batched encode then raises ``ValueError``. The FAQ fix
    is ``tokenizer.pad_token = tokenizer.eos_token`` — applied here (mutates
    the model's tokenizer in place, matching upstream guidance).
    """
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _render_prompt(tokenizer, prompt):
    """Render one prompt to text: ``str`` passes through (raw branch),
    a list of ``{role, content}`` messages goes through the chat template."""
    if isinstance(prompt, str):
        return prompt
    # Chat-template branch: list of message dicts, e.g.
    # [{"role": "user", "content": "Say hi in five words."}].
    return tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )


def generate_text(model, prompts, max_new_tokens=8, use_streamer=False):
    """Generate one completion per prompt; return ``list[str]``.

    - MLX branch: each prompt is encoded to numpy token ids
      (``tok(text, return_tensors="np")["input_ids"]``) and passed
      *positionally* — ``str`` prompts would raise
      ``ValueError: Invalid type str``. ``use_streamer`` is accepted but has
      no effect here (MLX swallows ``**kwargs``; passing a streamer would be
      silently ignored, so it is not passed — documented, not faked).
    - Generic branch: HF delegation; ``use_streamer=True`` attaches a
      ``transformers.TextStreamer`` so tokens stream to stdout.
    """
    tokenizer = _ensure_pad_token(_tokenizer_for(model))
    is_mlx = type(model).__name__ == _MLX_CLASS_NAME

    streamer = None
    if use_streamer and not is_mlx:
        from transformers import TextStreamer  # noqa: PLC0415 - only needed for streaming

        streamer = TextStreamer(tokenizer, skip_prompt=True)

    texts = []
    for prompt in prompts:
        text = _render_prompt(tokenizer, prompt)
        if is_mlx:
            # MUST be ids, never str (see module docstring): mirrors the
            # Step 1.4U bench fix, proven live at 0.3-0.5 tok/s on TinyLlama.
            ids = tokenizer(text, return_tensors="np")["input_ids"]
            out = model.generate(ids, max_new_tokens=max_new_tokens)
            texts.append(out if isinstance(out, str) else str(out))
        else:
            encoded = tokenizer(text, return_tensors="pt")
            gen_kwargs = {"max_new_tokens": max_new_tokens}
            if streamer is not None:
                gen_kwargs["streamer"] = streamer
            out = model.generate(encoded["input_ids"], **gen_kwargs)
            texts.append(tokenizer.decode(out[0], skip_special_tokens=True))
    return texts
