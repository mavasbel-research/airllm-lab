"""Shared fakes for airllm-lab tests (Steps 2.2).

Local fakes only — never real weights, never the network. Unit tests use
these with zero I/O; component tests add ``tmp`` workdirs via the
``workdir`` fixture in ``conftest.py``.
"""

from __future__ import annotations


class FakeTokenizer:
    """Minimal tokenizer double for :mod:`airllm_lab.generate`."""

    def __init__(self, pad_token="PAD", eos_token="EOS", chat_exc=None):
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.chat_exc = chat_exc
        self.calls = []
        self.chat_calls = []

    def __call__(self, text, return_tensors=None):
        self.calls.append((text, return_tensors))
        if return_tensors == "np":
            return {"input_ids": FakeIds(text)}
        return {"input_ids": [[1, 2, 3]]}

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        self.chat_calls.append((messages, tokenize, add_generation_prompt))
        if self.chat_exc is not None:
            raise self.chat_exc
        return "CHAT:" + "|".join(m["content"] for m in messages)

    def decode(self, ids, skip_special_tokens=True):
        return "decoded:%s" % (list(ids),)


class FakeIds:
    """Opaque token-id payload (MLX branch passes it through positionally)."""

    def __init__(self, text):
        self.text = text


class AirLLMLlamaMlx:
    """Fake macOS MLX backend.

    The class name is the wrapper's only source-verified discriminator
    (``generate._MLX_CLASS_NAME``), so it must match exactly.
    """

    def __init__(self, tokenizer=None, out="mlx text"):
        self.tokenizer = tokenizer or FakeTokenizer()
        self.out = out
        self.generate_calls = []

    def generate(self, x, temperature=0, max_new_tokens=None, **kwargs):
        self.generate_calls.append((x, max_new_tokens, kwargs))
        return self.out


class FakeGenericModel:
    """Fake generic (transformers-delegating) backend."""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or FakeTokenizer()
        self.generate_calls = []

    def generate(self, input_ids, **kwargs):
        self.generate_calls.append((input_ids, kwargs))
        return [[1, 2, 3, 4]]


class FakeAutoModel:
    """Fake ``airllm.AutoModel``: records loads, returns a canned model."""

    loads = []
    model = None
    error = None

    @classmethod
    def reset(cls, model=None, error=None):
        cls.loads = []
        cls.model = model
        cls.error = error

    @classmethod
    def from_pretrained(cls, hub_id, **kwargs):
        cls.loads.append((hub_id, kwargs))
        if cls.error is not None:
            raise cls.error
        return cls.model


class FakeStreamer:
    """Double for ``transformers.TextStreamer`` (records constructor args)."""

    instances = []

    def __init__(self, tokenizer, skip_prompt=True):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        FakeStreamer.instances.append(self)


class FakeCuda:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class FakeTorch:
    def __init__(self, available):
        self.cuda = FakeCuda(available)


class FakeCompleted:
    def __init__(self, returncode):
        self.returncode = returncode
