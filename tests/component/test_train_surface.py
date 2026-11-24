"""Component suite: train feature (local fakes only, no external dependency).

Feature: train. ``train_surface`` with a stubbed torch layer: ``--help``
always works; anything else is CUDA-gated with the verbatim upstream reason.
"""

import sys

import pytest

from airllm_lab.train import CUDA_GATE_MESSAGE, train_surface

from fakes import FakeTorch

FEATURE = "train"


def _no_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch(False))


@pytest.mark.scenario("train-surface")
@pytest.mark.dep("train-cli-help")
def test_train_surface_help(capsys):
    """train-surface: --help prints usage and returns 0 (no device needed)."""
    assert train_surface(["--help"]) == 0
    assert "train_surface" in capsys.readouterr().out


@pytest.mark.scenario("train-surface")
@pytest.mark.dep("train-blocked-mac")
@pytest.mark.negative("host-blocked training returns nonzero with verbatim reason")
def test_train_surface_blocked_mac(monkeypatch, capsys):
    """train-surface negative: smoke without CUDA is BLOCKED, never faked."""
    _no_cuda(monkeypatch)
    assert train_surface(["--steps", "1"]) == 1
    err = capsys.readouterr().err
    assert CUDA_GATE_MESSAGE in err
    assert "no training was run" in err
