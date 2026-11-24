"""Unit tests: package surface (``airllm_lab/__init__.py``). In-process, no I/O."""

import airllm_lab


def test_package_exports_all_names():
    assert set(airllm_lab.__all__) == {
        "DEFAULT_DEVICE",
        "default_shard_path",
        "generate_text",
        "load_model",
        "train_surface",
    }


def test_default_device_is_cpu():
    assert airllm_lab.DEFAULT_DEVICE == "cpu"


def test_missing_export_raises_negative():
    import pytest as _pytest

    with _pytest.raises(AttributeError):
        airllm_lab.no_such_export
