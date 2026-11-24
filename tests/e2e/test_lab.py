"""E2E suite: whole stack via the composition root (local fakes, runs last).

Feature coverage: every non-notebook use case from the plan table, wired
end to end as ``load_model -> generate_text -> train_surface`` through the
``airllm_lab`` package root. Notebook use cases (``nb-*``) are covered by
the plan mapping checked in ``scripts/check_granularity.py``, not here.
"""

import sys

import pytest

import airllm_lab
import airllm_lab.train as train_mod
from airllm_lab.train import train_surface

from fakes import (
    AirLLMLlamaMlx,
    FakeAutoModel,
    FakeCompleted,
    FakeGenericModel,
    FakeTorch,
)

PRIMARY = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

MAC_ERROR = (
    "Qwen2ForCausalLM is not supported on macOS yet. The macOS backend (MLX) "
    "only runs Llama-style models: LlamaForCausalLM, MistralForCausalLM."
)


@pytest.fixture()
def stack(monkeypatch):
    """Compose the whole stack on fakes: loader + CPU torch."""
    import airllm

    FakeAutoModel.reset(model=FakeGenericModel())
    monkeypatch.setattr(airllm, "AutoModel", FakeAutoModel)
    monkeypatch.delenv("AIRLLM_LAB_SHARDS", raising=False)
    monkeypatch.setitem(sys.modules, "torch", FakeTorch(False))
    return FakeAutoModel


def _load_generate(hub_id=PRIMARY, prompts=("Hello world",), **kwargs):
    model = airllm_lab.load_model(hub_id, device="cpu", **kwargs)
    return model, airllm_lab.generate_text(model, list(prompts), max_new_tokens=8)


@pytest.mark.scenario("bench-tiny-cpu")
def test_e2e_bench_tiny_cpu(stack):
    """bench-tiny-cpu e2e: load (CPU) -> generate (<=8 tokens) -> one row."""
    _, texts = _load_generate()
    assert texts == ["decoded:[1, 2, 3, 4]"]
    assert stack.loads[0][1]["device"] == "cpu"


@pytest.mark.scenario("bench-tiny-cpu")
@pytest.mark.negative("bogus id fails the stack load instead of yielding a row")
def test_e2e_bench_tiny_cpu_negative(stack):
    """bench-tiny-cpu e2e negative: bad id aborts before generation."""
    stack.error = OSError("offline: cannot reach hub")
    with pytest.raises(OSError, match="offline"):
        _load_generate(hub_id="no/such-model-xyz")


@pytest.mark.scenario("bench-compression-matrix")
def test_e2e_bench_compression_matrix(stack):
    """bench-compression-matrix e2e: None loads; 4bit is BLOCKED verbatim."""
    _, ok_texts = _load_generate(compression=None)
    assert ok_texts == ["decoded:[1, 2, 3, 4]"]
    stack.error = RuntimeError("bitsandbytes has no macOS wheel")
    with pytest.raises(RuntimeError, match="no macOS wheel"):
        _load_generate(compression="4bit")


@pytest.mark.scenario("bench-compression-matrix")
@pytest.mark.negative("unknown compression aborts that cell only")
def test_e2e_bench_compression_matrix_negative(stack):
    """bench-compression-matrix e2e negative: 2bit cell BLOCKED, reason kept."""
    stack.error = ValueError("compression=2bit unknown")
    with pytest.raises(ValueError, match="compression=2bit unknown"):
        _load_generate(compression="2bit")


@pytest.mark.scenario("infer-generate-cpu")
def test_e2e_infer_generate_cpu(stack):
    """infer-generate-cpu e2e: raw prompt through the full stack."""
    _, texts = _load_generate()
    assert len(texts) == 1 and isinstance(texts[0], str)


@pytest.mark.scenario("infer-generate-cpu")
@pytest.mark.negative("empty prompts short-circuit the whole stack")
def test_e2e_infer_generate_cpu_negative(stack):
    """infer-generate-cpu e2e negative: [] in -> [] out, no generation."""
    _, texts = _load_generate(prompts=[])
    assert texts == []


@pytest.mark.scenario("infer-chat-template")
def test_e2e_infer_chat_template(stack):
    """infer-chat-template e2e: messages render then generate."""
    messages = [{"role": "user", "content": "Say hi in five words."}]
    _, texts = _load_generate(prompts=[messages])
    assert texts == ["decoded:[1, 2, 3, 4]"]


@pytest.mark.scenario("infer-chat-template")
@pytest.mark.negative("template failure aborts generation with the verbatim error")
def test_e2e_infer_chat_template_negative(stack):
    """infer-chat-template e2e negative: broken template -> verbatim error."""
    from fakes import FakeTokenizer

    stack.model.tokenizer = FakeTokenizer(chat_exc=RuntimeError("no template"))
    with pytest.raises(RuntimeError, match="no template"):
        _load_generate(prompts=[[ {"role": "user", "content": "hi"} ]])


@pytest.mark.scenario("infer-profiling")
def test_e2e_infer_profiling(stack):
    """infer-profiling e2e: flags flow load -> generate end to end."""
    _, texts = _load_generate(profiling_mode=True, prefetching=False)
    assert texts == ["decoded:[1, 2, 3, 4]"]
    assert stack.loads[0][1]["profiling_mode"] is True


@pytest.mark.scenario("infer-profiling")
@pytest.mark.negative("rejected profiling flag aborts the stack verbatim")
def test_e2e_infer_profiling_negative(stack):
    """infer-profiling e2e negative: loader rejection propagates."""
    stack.error = TypeError("profiling_mode not supported")
    with pytest.raises(TypeError, match="profiling_mode not supported"):
        _load_generate(profiling_mode=True)


@pytest.mark.scenario("infer-negative-mac-error")
@pytest.mark.negative("non-Llama arch fails pre-download across the whole stack")
def test_e2e_infer_negative_mac_error(stack):
    """infer-negative-mac-error e2e: Qwen tiny blocked with the verbatim error."""
    stack.error = ValueError(MAC_ERROR)
    with pytest.raises(ValueError, match="not supported on macOS"):
        _load_generate(hub_id="Qwen/Qwen2-0.5B")


@pytest.mark.scenario("train-surface")
def test_e2e_train_surface_help():
    """train-surface e2e: --help through the composed package root."""
    assert airllm_lab.train_surface(["--help"]) == 0


@pytest.mark.scenario("train-surface")
@pytest.mark.negative("smoke without CUDA blocked end to end, exit 1")
def test_e2e_train_surface_blocked(stack, capsys):
    """train-surface e2e negative: host-blocked smoke, verbatim reason."""
    assert train_surface(["--steps", "1"]) == 1
    assert train_mod.CUDA_GATE_MESSAGE in capsys.readouterr().err


def test_e2e_mlx_backend_end_to_end():
    """e2e shape proof: MLX ids path returns text through the package root."""
    model = AirLLMLlamaMlx(out="tiny text")
    assert isinstance(airllm_lab.generate_text(model, ["hi"], max_new_tokens=8)[0], str)


def test_e2e_trainer_passthrough_shape(monkeypatch):
    """e2e shape proof: trainer subprocess return code passes through."""
    import subprocess

    monkeypatch.setitem(sys.modules, "torch", FakeTorch(True))
    monkeypatch.setattr(train_mod, "_locate_trainer", lambda: "some.trainer")
    monkeypatch.setattr(subprocess, "run", lambda cmd, check=False: FakeCompleted(0))
    assert train_surface(["--steps", "1"]) == 0
