"""Training surface: expose AirLLM's LoRA training entry points honestly.

Step 1.2 source truth (airllm v4.0.1, SHA 291a93ab45effeb5c2c0353ca0be2f690af7a616):

- Upstream training surface = two CLI scripts,
  ``air_llm/examples/train_qwen38_lora.py`` (Qwen3.8-27B, flags ``--data``
  ``--model`` ``--device`` ``--seq-len`` ``--epochs`` ``--steps`` ``--lora-r``
  ``--lora-alpha`` ``--lr`` ``--ce-chunk-size`` ``--max-seq-len``
  ``--save-adapter`` ``--verbose``) and
  ``air_llm/examples/train_qwen38_flash_next_lora.py`` (same plus
  ``--packed-expert-lora`` ``--keep-original`` ``--log-csv``).
- Hard gate (``train_qwen38_lora.py:50-52``): training is CUDA-only —
  ``if not torch.cuda.is_available(): print("CUDA is required for this
  trainer."); sys.exit(1)``. On this macOS host (``torch.cuda.is_available()``
  is ``False``) training is host-blocked: this module reports that verbatim
  and returns non-zero. It NEVER fakes training.
- Gap (recorded, not guessed): the published ``airllm==4.0.1`` wheel does not
  ship the ``examples/`` scripts, so there is no in-venv trainer CLI to exec.
  The smoke path below resolves the trainer via import and reports its
  absence verbatim instead of pretending to train.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

#: Trainer module candidates, in preference order. Neither is shipped in the
#: ``airllm==4.0.1`` wheel (verified: no ``examples/`` under site-packages) —
#: the first importable one wins; when none import, the smoke path reports
#: the absence verbatim (see :func:`train_surface`).
_TRAINER_CANDIDATES = (
    "airllm.examples.train_qwen38_lora",
    "airllm.examples.train_qwen38_flash_next_lora",
)

#: Verbatim upstream CUDA gate message (``train_qwen38_lora.py:50-52``).
CUDA_GATE_MESSAGE = "CUDA is required for this trainer."

USAGE = """usage: train_surface [--help] [--steps N] [trainer args ...]

Thin surface over AirLLM streamed LoRA training (Step 1.2 truth: upstream
flags --data --model --device --seq-len --epochs --steps --lora-r --lora-alpha
--lr --ce-chunk-size --max-seq-len --save-adapter --verbose).

  --help     show this message (no training, always works)
  --steps N  smoke run: forward N trainer steps, then stop

Training is CUDA-gated upstream: without CUDA this command prints the
verbatim reason and exits 1. It never fakes training.
"""


def _locate_trainer():
    """Return the first importable trainer module name, or ``None``."""
    for name in _TRAINER_CANDIDATES:
        if importlib.util.find_spec(name) is not None:
            return name
    return None


def train_surface(argv=None):
    """Expose the training CLI surface; return the process exit code as int.

    - ``--help``/``-h``: print usage, return 0 (no device needed).
    - anything else (e.g. ``--steps`` smoke): first apply the upstream
      CUDA gate — without CUDA, print the verbatim reason and return 1.
      On a CUDA host, exec the resolved trainer module with the given args
      and return its exit code; when no trainer module ships in the
      installed wheel, print that verbatim and return 2.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if any(a in ("-h", "--help") for a in args) or not args:
        print(USAGE)
        return 0

    import torch  # noqa: PLC0415 - deferred so --help stays dependency-light

    if not torch.cuda.is_available():
        print(
            "train_surface: BLOCKED: training is CUDA-gated upstream "
            f"(train_qwen38_lora.py:50-52): {CUDA_GATE_MESSAGE} "
            f"(torch.cuda.is_available() is {torch.cuda.is_available()}); "
            "no training was run.",
            file=sys.stderr,
        )
        return 1

    trainer = _locate_trainer()
    if trainer is None:
        print(
            "train_surface: BLOCKED: no trainer CLI ships in the installed "
            f"airllm wheel (looked for {', '.join(_TRAINER_CANDIDATES)}); "
            "no training was run.",
            file=sys.stderr,
        )
        return 2

    completed = subprocess.run(
        [sys.executable, "-m", trainer, *args],
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":  # pragma: no cover - thin passthrough
    sys.exit(train_surface())
