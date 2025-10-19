"""
SOTA model implementations for competitive mining
"""

# Only import what we're using to avoid dependency issues
# from .flux_generator import FluxImageGenerator
from .sdxl_turbo_generator import SDXLTurboGenerator
from .background_remover import SOTABackgroundRemover

__all__ = ["SDXLTurboGenerator", "SOTABackgroundRemover"]
