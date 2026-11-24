"""Unit tests: ``airllm_lab.generate``. In-process, no I/O (models faked)."""

import pytest

from airllm_lab.generate import (
    _ensure_pad_token,
    _render_prompt,
    _tokenizer_for,
    generate_text,
)

from fakes import (
    AirLLMLlamaMlx,
    FakeGenericModel,
    FakeStreamer,
    FakeTokenizer,
)


def test_tokenizer_for_returns_bundled_tokenizer():
    tok, model = FakeTokenizer(), FakeGenericModel()
    model.tokenizer = tok
    assert _tokenizer_for(model) is tok


def test_ensure_pad_token_falls_back_to_eos():
    tok = FakeTokenizer(pad_token=None, eos_token="EOS")
    out = _ensure_pad_token(tok)
    assert out.pad_token == "EOS"


def test_ensure_pad_token_kept_when_present():
    tok = FakeTokenizer(pad_token="PAD", eos_token="EOS")
    assert _ensure_pad_token(tok).pad_token == "PAD"


def test_ensure_pad_token_missing_attr_negative():
    class Bare:
        eos_token = "EOS"

    tok = _ensure_pad_token(Bare())
    assert tok.pad_token == "EOS"


def test_render_prompt_str_passthrough():
    tok = FakeTokenizer()
    assert _render_prompt(tok, "hello") == "hello"


def test_render_prompt_chat_template_branch():
    tok = FakeTokenizer()
    messages = [{"role": "user", "content": "Say hi."}]
    assert _render_prompt(tok, messages) == "CHAT:Say hi."
    ((got, tokenize, add_g),) = tok.chat_calls
    assert got == messages and tokenize is False and add_g is True


def test_generic_raw_generate_cpu():
    model = FakeGenericModel()
    assert generate_text(model, ["Hello world"], max_new_tokens=8) == ["decoded:[1, 2, 3, 4]"]
    ((ids, kwargs),) = model.generate_calls
    assert ids == [[1, 2, 3]] and kwargs == {"max_new_tokens": 8}


def test_generic_no_streamer_by_default():
    model = FakeGenericModel()
    generate_text(model, ["hi"], use_streamer=False)
    assert model.generate_calls[0][1] == {"max_new_tokens": 8}


def test_generic_streamer_attached(monkeypatch):
    import transformers

    FakeStreamer.instances.clear()
    monkeypatch.setattr(transformers, "TextStreamer", FakeStreamer)
    model = FakeGenericModel()
    generate_text(model, ["hi"], max_new_tokens=8, use_streamer=True)
    assert len(FakeStreamer.instances) == 1
    assert FakeStreamer.instances[0].skip_prompt is True
    assert model.generate_calls[0][1]["streamer"] is FakeStreamer.instances[0]


def test_mlx_ids_branch_positional():
    tok = FakeTokenizer()
    model = AirLLMLlamaMlx(tokenizer=tok, out="mlx text")
    assert generate_text(model, ["Hello world"], max_new_tokens=8) == ["mlx text"]
    ((ids, max_new, kwargs),) = model.generate_calls
    assert isinstance(ids, object) and not isinstance(ids, str)
    assert ids.text == "Hello world" and max_new == 8 and kwargs == {}
    assert tok.calls[0] == ("Hello world", "np")


def test_mlx_nonstring_output_coerced_to_str():
    model = AirLLMLlamaMlx(out=["tok1", "tok2"])
    assert generate_text(model, ["hi"]) == ["['tok1', 'tok2']"]


def test_mlx_streamer_ignored(monkeypatch):
    import transformers

    FakeStreamer.instances.clear()
    monkeypatch.setattr(transformers, "TextStreamer", FakeStreamer)
    model = AirLLMLlamaMlx(out="mlx text")
    assert generate_text(model, ["hi"], use_streamer=True) == ["mlx text"]
    assert FakeStreamer.instances == []
    assert model.generate_calls[0][2] == {}


def test_chat_template_generate_generic():
    model = FakeGenericModel()
    out = generate_text(model, [[{"role": "user", "content": "Say hi."}]])
    assert out == ["decoded:[1, 2, 3, 4]"]
    assert model.tokenizer.chat_calls[0][0] == [{"role": "user", "content": "Say hi."}]


def test_empty_prompts_returns_empty_list():
    assert generate_text(FakeGenericModel(), []) == []


def test_multiple_prompts_keep_order():
    model = FakeGenericModel()
    assert generate_text(model, ["a", "b"]) == ["decoded:[1, 2, 3, 4]"] * 2
