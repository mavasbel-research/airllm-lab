"""Live suite: airllm-shards dependency (endpoint-gated, tiny IDs only).

Dependency: airllm-shards. Use case: shard-split-reuse.
The full load runs only with ``AIRLLM_LAB_LIVE_SHARDS=1`` (it downloads the
PRIMARY tiny weights once, then reuses shards); otherwise it skips with a
recorded reason. The macOS-architecture negative is cheap: AirLLM raises
pre-download, so no weights move.
"""

import os

import pytest

PRIMARY = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
NEGATIVE = "Qwen/Qwen2-0.5B"

FEATURE = "live-shards"


def _skip_offline(exc):
    pytest.skip("offline or Hub unreachable, shard-split-reuse not proven: %r" % (exc,))


@pytest.mark.scenario("shard-split-reuse")
@pytest.mark.dep("shard-split-reuse")
def test_shard_split_reuse(workdir):
    """shard-split-reuse: second load of PRIMARY reuses the same shard dir."""
    if os.environ.get("AIRLLM_LAB_LIVE_SHARDS") != "1":
        pytest.skip(
            "live shard load disabled (set AIRLLM_LAB_LIVE_SHARDS=1 to fetch "
            "PRIMARY tiny once); shard-split-reuse not proven here"
        )
    import airllm_lab.load as load_mod
    from airllm_lab.load import load_model

    monkey_shards = workdir / "shards"
    os.environ["AIRLLM_LAB_SHARDS"] = str(monkey_shards)
    try:
        first = load_model(PRIMARY, device="cpu")
        second = load_model(PRIMARY, device="cpu")
    except Exception as exc:  # noqa: BLE001 - offline shows up here
        _skip_offline(exc)
    finally:
        os.environ.pop("AIRLLM_LAB_SHARDS", None)
    assert load_mod.default_shard_path(PRIMARY).parent == monkey_shards
    assert type(first) is type(second)


@pytest.mark.scenario("shard-split-reuse")
@pytest.mark.dep("shard-split-reuse")
@pytest.mark.negative("non-Llama arch raises the macOS ValueError pre-download")
def test_shard_negative_arch_pre_download():
    """shard-split-reuse negative: Qwen tiny raises before any shard is written."""
    from airllm_lab.load import load_model

    try:
        load_model(NEGATIVE, device="cpu")
    except ValueError as exc:
        assert "macOS" in str(exc)
        return
    except Exception as exc:  # noqa: BLE001 - offline shows up here
        _skip_offline(exc)
    raise AssertionError("Qwen tiny unexpectedly loaded on macOS")
