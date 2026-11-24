"""Unit tests: ``airllm_lab.load``. In-process, no I/O (AirLLM stubbed)."""

from pathlib import Path

import pytest

import airllm_lab.load as load_mod  # noqa: F401 - ensures module import path under test
from airllm_lab.load import (
    DEFAULT_DEVICE,
    SHARDS_DIRNAME,
    SHARDS_ENV_VAR,
    default_shard_path,
    load_model,
)

from fakes import FakeAutoModel, FakeGenericModel


@pytest.fixture()
def fake_automodel(monkeypatch):
    import airllm

    model = FakeGenericModel()
    FakeAutoModel.reset(model=model)
    monkeypatch.setattr(airllm, "AutoModel", FakeAutoModel)
    return model


def test_default_device_is_cpu():
    assert DEFAULT_DEVICE == "cpu"


def test_shard_constants_match_gitignore():
    assert SHARDS_ENV_VAR == "AIRLLM_LAB_SHARDS"
    assert SHARDS_DIRNAME == "airllm-shards"


def test_default_shard_path_repo_ignored_dir(monkeypatch):
    monkeypatch.delenv(SHARDS_ENV_VAR, raising=False)
    assert default_shard_path("owner/repo") == Path.cwd() / "airllm-shards" / "owner__repo"


def test_default_shard_path_env_override_branch(monkeypatch):
    monkeypatch.setenv(SHARDS_ENV_VAR, "/scratch/shards")
    assert default_shard_path("owner/repo") == Path("/scratch/shards/owner__repo")


def test_default_shard_path_slug_replaces_every_slash(monkeypatch):
    monkeypatch.delenv(SHARDS_ENV_VAR, raising=False)
    assert default_shard_path("a/b/c").name == "a__b__c"


def test_load_model_applies_cpu_and_shard_defaults(fake_automodel, monkeypatch):
    monkeypatch.delenv(SHARDS_ENV_VAR, raising=False)
    out = load_model("owner/repo")
    assert out is fake_automodel
    (hub_id, kwargs), = FakeAutoModel.loads
    assert hub_id == "owner/repo"
    assert kwargs["device"] == "cpu"
    assert kwargs["layer_shards_saving_path"] == str(default_shard_path("owner/repo"))


def test_load_model_device_override(fake_automodel):
    load_model("owner/repo", device="cuda:0")
    (_, kwargs), = FakeAutoModel.loads
    assert kwargs["device"] == "cuda:0"


def test_load_model_shard_override(fake_automodel):
    load_model("owner/repo", layer_shards_saving_path="/else/where")
    (_, kwargs), = FakeAutoModel.loads
    assert kwargs["layer_shards_saving_path"] == "/else/where"


def test_load_model_passthrough_extra_kwargs(fake_automodel):
    load_model("owner/repo", compression=None, profiling_mode=True, prefetching=False)
    (_, kwargs), = FakeAutoModel.loads
    assert kwargs["compression"] is None
    assert kwargs["profiling_mode"] is True
    assert kwargs["prefetching"] is False


def test_load_model_propagates_loader_error_negative(fake_automodel):
    FakeAutoModel.error = ValueError("Qwen2ForCausalLM is not supported on macOS yet.")
    with pytest.raises(ValueError, match="not supported on macOS"):
        load_model("Qwen/Qwen2-0.5B")
