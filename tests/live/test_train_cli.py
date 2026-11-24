"""Live suite: airllm-train dependency (real torch gate, no fakes).

Dependency: airllm-train. Use cases: train-cli-help, train-blocked-mac.
``--help`` always runs; the smoke path is CUDA-gated on the real host torch
(and skips with a recorded reason on a CUDA host instead of asserting blocked).
"""

import pytest

from airllm_lab.train import CUDA_GATE_MESSAGE, train_surface

FEATURE = "live-train"


@pytest.mark.scenario("train-cli-help")
@pytest.mark.dep("train-cli-help")
def test_train_cli_help(capsys):
    """train-cli-help: --help exits 0 without any device."""
    assert train_surface(["--help"]) == 0
    assert "usage" in capsys.readouterr().out.lower()


@pytest.mark.scenario("train-blocked-mac")
@pytest.mark.dep("train-blocked-mac")
@pytest.mark.negative("CUDA-gated smoke is BLOCKED on this host, never faked")
def test_train_blocked_mac(capsys):
    """train-blocked-mac: smoke without CUDA returns 1 with the verbatim reason."""
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA host: train-blocked-mac does not apply here")
    assert train_surface(["--steps", "1"]) == 1
    assert CUDA_GATE_MESSAGE in capsys.readouterr().err


@pytest.mark.scenario("train-cli-help")
@pytest.mark.dep("train-cli-help")
@pytest.mark.negative("unrecognized smoke args stay CUDA-gated, never train")
def test_train_cli_unknown_args_gated(capsys):
    """train-cli-help negative: non-help args hit the CUDA gate first."""
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA host: gate does not apply here")
    assert train_surface(["--bogus-flag"]) == 1
    assert "no training was run" in capsys.readouterr().err
