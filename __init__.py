"""Top-level package for latent_shift."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """Yosshi999"""
__email__ = "Yosshi999@users.noreply.github.com"
__version__ = "0.0.1"

from .src.latent_shift.nodes import NODE_CLASS_MAPPINGS
from .src.latent_shift.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
