"""airllm_lab: thin reusable wrapper over AirLLM (plan Step 2.1).

Functions only (KISS): :func:`load_model`, :func:`generate_text`,
:func:`train_surface`. No classes — AirLLM owns every seam; each seam the
wrapper crosses is justified in a comment at the crossing point.
"""

from airllm_lab.generate import generate_text
from airllm_lab.load import DEFAULT_DEVICE, default_shard_path, load_model
from airllm_lab.train import train_surface

__all__ = [
    "DEFAULT_DEVICE",
    "default_shard_path",
    "generate_text",
    "load_model",
    "train_surface",
]
