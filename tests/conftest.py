"""Pytest fixtures for airllm-lab tests (Step 2.2).

All temporary artifacts go under the plan scratch dir
(``.../airllm-lab-plan/testtiers/``) — never the repo, never /tmp directly.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

SCRATCH_ROOT = Path(
    os.environ.get(
        "AIRLLM_LAB_TESTTIERS",
        "/var/folders/_m/jyv5nmsn4y305mjjcm_4fq240000gn/T/opencode/ses_df5b5e7cbffeUTbz4f1XoUo1Q3/airllm-lab-plan/testtiers",
    )
)


@pytest.fixture()
def workdir():
    """Function-scoped temp dir under the plan scratch root (auto-removed)."""
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="t_", dir=str(SCRATCH_ROOT)))
    yield path
    shutil.rmtree(path, ignore_errors=True)
