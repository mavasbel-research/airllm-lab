"""Unit tests: ``airllm_lab.train``. In-process, no I/O (torch faked)."""

import subprocess
import sys

import pytest

import airllm_lab.train as train_mod
from airllm_lab.train import (
    CUDA_GATE_MESSAGE,
    USAGE,
    _locate_trainer,
    train_surface,
)

from fakes import FakeCompleted, FakeTorch


def _fake_torch(monkeypatch, available):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch(available))


def test_help_returns_zero(capsys):
    assert train_surface(["--help"]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_h_flag_returns_zero(capsys):
    assert train_surface(["-h"]) == 0
    assert USAGE.splitlines()[0] in capsys.readouterr().out


def test_no_args_returns_zero():
    assert train_surface([]) == 0


def test_argv_none_reads_sys_argv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_surface", "--help"])
    assert train_surface() == 0


def test_cuda_blocked_returns_one_with_verbatim_reason(monkeypatch, capsys):
    _fake_torch(monkeypatch, False)
    assert train_surface(["--steps", "1"]) == 1
    err = capsys.readouterr().err
    assert CUDA_GATE_MESSAGE in err
    assert "torch.cuda.is_available() is False" in err


def test_locate_trainer_first_found(monkeypatch):
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda name: object() if name == train_mod._TRAINER_CANDIDATES[0] else None
    )
    assert _locate_trainer() == train_mod._TRAINER_CANDIDATES[0]


def test_locate_trainer_second_found(monkeypatch):
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: object() if name == train_mod._TRAINER_CANDIDATES[1] else None,
    )
    assert _locate_trainer() == train_mod._TRAINER_CANDIDATES[1]


def test_locate_trainer_none_found(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert _locate_trainer() is None


def test_cuda_host_without_trainer_returns_two_negative(monkeypatch, capsys):
    _fake_torch(monkeypatch, True)
    monkeypatch.setattr(train_mod, "_locate_trainer", lambda: None)
    assert train_surface(["--steps", "1"]) == 2
    assert "no trainer CLI ships" in capsys.readouterr().err


def test_cuda_host_runs_trainer_and_returns_its_code_negative(monkeypatch):
    _fake_torch(monkeypatch, True)
    monkeypatch.setattr(train_mod, "_locate_trainer", lambda: "some.trainer")
    seen = {}

    def fake_run(cmd, check=False):
        seen["cmd"] = cmd
        return FakeCompleted(3)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert train_surface(["--steps", "2", "--lr", "1e-4"]) == 3
    assert seen["cmd"] == [sys.executable, "-m", "some.trainer", "--steps", "2", "--lr", "1e-4"]
