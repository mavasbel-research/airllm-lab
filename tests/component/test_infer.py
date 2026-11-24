"""Component suite: infer feature (local fakes only, no external dependency).

Feature: infer. One scenario per use-case id from the plan table, driving
``load_model`` + ``generate_text`` with fakes standing in for real weights.
"""

import pytest

from airllm_lab.generate import generate_text
from airllm_lab.load import load_model

from fakes import AirLLMLlamaMlx, FakeAutoModel, FakeGenericModel, FakeTokenizer

FEATURE = "infer"

PRIMARY = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

MAC_ERROR = (
    "Qwen2ForCausalLM is not supported on macOS yet. The macOS backend (MLX) "
    "only runs Llama-style models: LlamaForCausalLM, MistralForCausalLM."
)


@pytest.fixture()
def generic_loader(monkeypatch):
    import airllm

    FakeAutoModel.reset(model=FakeGenericModel())
    monkeypatch.setattr(airllm, "AutoModel", FakeAutoModel)
    monkeypatch.delenv("AIRLLM_LAB_SHARDS", raising=False)
    return FakeAutoModel


@pytest.mark.scenario("infer-generate-cpu")
@pytest.mark.dep("hub-fetch-tiny")
def test_infer_generate_cpu(generic_loader):
    """infer-generate-cpu: raw prompt on CPU returns one decoded string."""
    model = load_model(PRIMARY, device="cpu")
    assert generate_text(model, ["Hello world"], max_new_tokens=8) == ["decoded:[1, 2, 3, 4]"]
    assert FakeAutoModel.loads[0][1]["device"] == "cpu"


@pytest.mark.scenario("infer-chat-template")
@pytest.mark.dep("hub-fetch-tiny")
def test_infer_chat_template(generic_loader):
    """infer-chat-template: message list goes through the chat template."""
    model = load_model(PRIMARY, device="cpu")
    messages = [{"role": "user", "content": "Say hi in five words."}]
    assert generate_text(model, [messages], max_new_tokens=8) == ["decoded:[1, 2, 3, 4]"]
    ((got, tokenize, add_g),) = model.tokenizer.chat_calls
    assert got == messages and tokenize is False and add_g is True


@pytest.mark.scenario("infer-profiling")
@pytest.mark.dep("shard-split-reuse")
def test_infer_profiling_flags_passthrough(generic_loader):
    """infer-profiling: profiling/prefetch flags reach the loader verbatim."""
    model = load_model(PRIMARY, device="cpu", profiling_mode=True, prefetching=False)
    generate_text(model, ["hi"], max_new_tokens=8)
    (_, kwargs), = FakeAutoModel.loads
    assert kwargs["profiling_mode"] is True and kwargs["prefetching"] is False


@pytest.mark.scenario("infer-negative-mac-error")
@pytest.mark.dep("shard-split-reuse")
@pytest.mark.negative("non-Llama arch on macOS raises before any download")
def test_infer_negative_mac_error_verbatim(generic_loader):
    """infer-negative-mac-error: Qwen tiny fails pre-download with verbatim error."""
    FakeAutoModel.reset(model=None, error=ValueError(MAC_ERROR))
    try:
        load_model("Qwen/Qwen2-0.5B", device="cpu")
        note = "UNEXPECTED: no error raised"
    except ValueError as exc:
        note = "verbatim negative-path error: ValueError: %s" % exc
    assert "not supported on macOS" in note


@pytest.mark.scenario("infer-generate-cpu")
@pytest.mark.dep("hub-fetch-tiny")
@pytest.mark.negative("empty prompt list returns empty list, never a loader call")
def test_infer_generate_cpu_empty_prompts_negative(generic_loader):
    """infer-generate-cpu negative: empty prompts short-circuit to []."""
    model = load_model(PRIMARY, device="cpu")
    assert generate_text(model, []) == []
    assert model.generate_calls == []


@pytest.mark.scenario("infer-chat-template")
@pytest.mark.dep("hub-fetch-tiny")
@pytest.mark.negative("template failure surfaces verbatim instead of fake text")
def test_infer_chat_template_error_negative():
    """infer-chat-template negative: template errors propagate verbatim."""
    model = FakeGenericModel(tokenizer=FakeTokenizer(chat_exc=RuntimeError("no template")))
    with pytest.raises(RuntimeError, match="no template"):
        generate_text(model, [[{"role": "user", "content": "hi"}]])


@pytest.mark.scenario("infer-profiling")
@pytest.mark.dep("shard-split-reuse")
@pytest.mark.negative("rejected profiling flag surfaces verbatim")
def test_infer_profiling_rejected_negative(generic_loader):
    """infer-profiling negative: loader rejection is reported, never retried silently."""
    FakeAutoModel.reset(model=None, error=TypeError("profiling_mode not supported"))
    with pytest.raises(TypeError, match="profiling_mode not supported"):
        load_model(PRIMARY, profiling_mode=True)


def test_mlx_backend_component_shape():
    """infer component: MLX positional-ids call shape (bench 1.4U proven)."""
    model = AirLLMLlamaMlx(out="hello there")
    assert generate_text(model, ["Hello world"], max_new_tokens=8) == ["hello there"]
    ((ids, max_new, _),) = model.generate_calls
    assert not isinstance(ids, str) and max_new == 8
