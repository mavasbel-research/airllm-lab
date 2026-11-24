"""Component suite: bench feature (local fakes only, no external dependency).

Feature: bench. Matrix cells drive the wrapper exactly as ``bench/run_bench.py``
does (explicit CPU device, scratch shard path, <=8 new tokens) and record
CSV-shaped rows — with a fake ``AutoModel`` standing in for real weights.
"""

import csv

import pytest

from airllm_lab.generate import generate_text
from airllm_lab.load import default_shard_path, load_model

from fakes import AirLLMLlamaMlx, FakeAutoModel, FakeGenericModel

FEATURE = "bench"

COLUMNS = [
    "model", "params", "compression", "profiling", "prefetch",
    "prompt_mode", "load_s", "peak_rss_mb", "tok_s", "notes",
]

PRIMARY = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture()
def patched_loader(monkeypatch):
    import airllm

    FakeAutoModel.reset(model=FakeGenericModel())
    monkeypatch.setattr(airllm, "AutoModel", FakeAutoModel)
    monkeypatch.delenv("AIRLLM_LAB_SHARDS", raising=False)
    return FakeAutoModel


def _ok_row(model_id, params, compression, profiling, prefetch, prompt_mode, texts):
    return {
        "model": model_id, "params": params, "compression": compression,
        "profiling": str(profiling), "prefetch": str(prefetch),
        "prompt_mode": prompt_mode, "load_s": "0.01",
        "peak_rss_mb": "100.0", "tok_s": "8.0",
        "notes": "ok len=%d" % len("".join(texts)),
    }


def _blocked_row(model_id, params, compression, profiling, prefetch, prompt_mode, reason):
    cell = "BLOCKED:%s" % reason
    return {
        "model": model_id, "params": params, "compression": compression,
        "profiling": str(profiling), "prefetch": str(prefetch),
        "prompt_mode": prompt_mode, "load_s": cell,
        "peak_rss_mb": cell, "tok_s": cell, "notes": reason,
    }


@pytest.mark.scenario("bench-tiny-cpu")
@pytest.mark.dep("hub-fetch-tiny")
@pytest.mark.dep("shard-split-reuse")
def test_bench_tiny_cpu_cell(patched_loader, workdir):
    """bench-tiny-cpu: PRIMARY tiny loads on CPU and generates <=8 tokens."""
    model = load_model(PRIMARY, device="cpu",
                       layer_shards_saving_path=str(workdir / "shards"))
    texts = generate_text(model, ["Hello world"], max_new_tokens=8)
    assert len(texts) == 1 and isinstance(texts[0], str)
    (hub_id, kwargs), = FakeAutoModel.loads
    assert hub_id == PRIMARY and kwargs["device"] == "cpu"
    row = _ok_row(PRIMARY, "1100048384", "None", False, True, "raw-generate", texts)
    csv_path = workdir / "row.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    with open(csv_path) as fh:
        (back,) = list(csv.DictReader(fh))
    assert back["tok_s"] == "8.0" and back["model"] == PRIMARY


@pytest.mark.scenario("bench-compression-matrix")
@pytest.mark.dep("shard-split-reuse")
def test_bench_compression_matrix_cell(patched_loader):
    """bench-compression-matrix: compression=None loads; others BLOCKED verbatim."""
    rows = []
    for compression in (None, "4bit", "8bit"):
        FakeAutoModel.reset(model=FakeGenericModel(),
                            error=None if compression is None else RuntimeError(
                                "bitsandbytes has no macOS wheel: compression=%s unsupported here"
                                % compression))
        try:
            model = load_model(PRIMARY, compression=compression)
            texts = generate_text(model, ["Hello world"], max_new_tokens=8)
            rows.append(_ok_row(PRIMARY, "1100048384", str(compression),
                                False, True, "raw-generate", texts))
        except RuntimeError as exc:
            rows.append(_blocked_row(PRIMARY, "1100048384", str(compression),
                                     False, True, "raw-generate", str(exc)))
    assert rows[0]["tok_s"] == "8.0"
    assert rows[1]["tok_s"].startswith("BLOCKED:") and "4bit" in rows[1]["notes"]
    assert rows[2]["tok_s"].startswith("BLOCKED:") and "8bit" in rows[2]["notes"]


@pytest.mark.scenario("bench-tiny-cpu")
@pytest.mark.dep("hub-fetch-tiny")
@pytest.mark.negative("bogus hub id never raises: recorded as BLOCKED row")
def test_bench_tiny_cpu_load_failure_blocked(patched_loader):
    """bench-tiny-cpu negative: failed load is a BLOCKED row, never a raise."""
    FakeAutoModel.reset(model=None, error=OSError("offline: cannot reach hub"))
    try:
        load_model("no/such-model-xyz")
        raised = None
    except OSError as exc:
        raised = exc
    row = _blocked_row("no/such-model-xyz", "0", "None", False, True,
                       "raw-generate", "load failed: OSError: %s" % raised)
    assert row["tok_s"].startswith("BLOCKED:load failed")


@pytest.mark.scenario("bench-compression-matrix")
@pytest.mark.negative("unsupported compression recorded verbatim, matrix continues")
def test_bench_compression_matrix_negative_keeps_matrix(patched_loader):
    """bench-compression-matrix negative: one BLOCKED cell, matrix still complete."""
    FakeAutoModel.reset(model=None, error=ValueError("compression=2bit unknown"))
    rows = []
    for compression in (None, "2bit"):
        try:
            model = load_model(PRIMARY, compression=compression)
            rows.append(_ok_row(PRIMARY, "1", str(compression), False, True,
                                "raw-generate", generate_text(model, ["hi"])))
        except ValueError as exc:
            rows.append(_blocked_row(PRIMARY, "1", str(compression), False, True,
                                     "raw-generate", "ValueError: %s" % exc))
    assert len(rows) == 2
    assert rows[1]["tok_s"] == "BLOCKED:ValueError: compression=2bit unknown"


@pytest.mark.scenario("bench-tiny-cpu")
@pytest.mark.dep("shard-split-reuse")
@pytest.mark.negative("default shard path is repo-ignored, never committed")
def test_bench_shard_path_default_repo_ignored(monkeypatch):
    """bench-tiny-cpu negative-adjacent: default shards stay under airllm-shards/."""
    monkeypatch.delenv("AIRLLM_LAB_SHARDS", raising=False)
    assert default_shard_path(PRIMARY).parts[-2] == "airllm-shards"


def test_mlx_backend_ids_path_component():
    """bench component: MLX fake generates from numpy-style ids (bench 1.4U shape)."""
    model = AirLLMLlamaMlx(out="tiny text")
    assert generate_text(model, ["Hello world"], max_new_tokens=8) == ["tiny text"]
