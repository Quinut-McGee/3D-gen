"""
SOTA model implementations for competitive mining
"""

from .flux_generator import FluxImageGenerator
from .background_remover import SOTABackgroundRemover

__all__ = ["FluxImageGenerator", "SOTABackgroundRemover"]
