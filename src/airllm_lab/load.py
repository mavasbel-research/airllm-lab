"""Load helper: one entry point for every AirLLM model load.

Step 1.2 source truth (airllm v4.0.1, SHA 291a93ab45effeb5c2c0353ca0be2f690af7a616):
``AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)`` takes no
defaults of its own — everything passes through to the concrete constructor.
"""

import os
from pathlib import Path

#: Step 1.2 device truth: the generic constructor defaults to ``device=None``
#: (-> ``default_device()`` picks cuda/mps/cpu), while the macOS MLX
#: constructor signature says ``device="cuda:0"`` but never uses it. Passing
#: CPU explicitly keeps a CPU-only host off the ``cuda:0`` path, which would
#: otherwise raise ``Torch not compiled with CUDA`` (bench/run_bench.py proves
#: this works: 9 numeric rows at ``device=cpu``).
DEFAULT_DEVICE = "cpu"

#: Name of the env var overriding where layer shards are written.
SHARDS_ENV_VAR = "AIRLLM_LAB_SHARDS"

#: Default shard directory name (relative to the caller's cwd). This exact
#: name is listed in the repo ``.gitignore`` (``airllm-shards/``), so default
#: loads never dirty the working tree.
SHARDS_DIRNAME = "airllm-shards"


def default_shard_path(hub_id):
    """Return the default ``layer_shards_saving_path`` for *hub_id*.

    ``$AIRLLM_LAB_SHARDS/<owner__repo>`` when the env var is set (tests and
    scratch runs point it at temp dirs), else ``./airllm-shards/<owner__repo>``
    (repo-ignored, never committed).
    """
    root = Path(os.environ.get(SHARDS_ENV_VAR, Path.cwd() / SHARDS_DIRNAME))
    return root / hub_id.replace("/", "__")


def load_model(hub_id, **airllm_kwargs):
    """Load *hub_id* via AirLLM and return the model.

    Seam (justification): every load is routed through ``AutoModel``
    (``airllm/auto_model.py``) instead of importing a concrete class. Using a
    concrete class directly for the wrong architecture fails (e.g. ``ValueError``
    from ``max()`` on an empty layer map, or silently wrong weights); the
    README FAQ prescribes ``AutoModel`` routing, and on darwin ``AutoModel``
    additionally enforces the macOS gate (``ValueError`` for non-Llama-style
    architectures before any download/split — the Step 1.4 negative path).

    Defaults applied (each overridable via ``**airllm_kwargs``):

    - ``device="cpu"`` (:data:`DEFAULT_DEVICE`, Step 1.2 truth above).
    - ``layer_shards_saving_path`` -> :func:`default_shard_path` (repo-ignored
      dir; callers needing scratch isolation set ``$AIRLLM_LAB_SHARDS``).
    """
    from airllm import AutoModel  # noqa: PLC0415 - deferred so import time stays free of torch/mlx

    kwargs = {
        "device": DEFAULT_DEVICE,
        "layer_shards_saving_path": str(default_shard_path(hub_id)),
        **airllm_kwargs,
    }
    return AutoModel.from_pretrained(hub_id, **kwargs)
