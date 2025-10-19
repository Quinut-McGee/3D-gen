"""
FLUX.1-schnell integration for ultra-fast, high-quality text-to-image generation.

FLUX.1-schnell is perfect for competitive mining:
- 1-4 steps for generation (vs 50+ for diffusion models)
- SOTA quality (2024)
- Free for commercial use
- Faster than MVDream while maintaining quality
"""

from diffusers import FluxPipeline
import torch
from PIL import Image
from typing import Optional, Union
from loguru import logger


class FluxImageGenerator:
    """
    FLUX.1-schnell text-to-image generator.

    This replaces MVDream for the initial image generation step.
    Much faster (1-4 steps) with comparable or better quality.
    """

    def __init__(
        self,
        device: str = "cuda",
        torch_dtype=torch.bfloat16,
        enable_optimization: bool = True
    ):
        """
        Initialize FLUX.1-schnell pipeline.

        Args:
            device: CUDA device
            torch_dtype: Data type (bfloat16 recommended for speed + quality)
            enable_optimization: Enable memory and speed optimizations
        """
        self.device = device
        self.torch_dtype = torch_dtype

        logger.info("Loading FLUX.1-schnell pipeline...")

        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch_dtype
        )

        self.pipe.to(device)

        # Optimizations for RTX 4090
        if enable_optimization:
            logger.info("Applying optimizations for RTX 4090...")

            # Enable memory efficient attention
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("✅ xFormers enabled")
            except Exception as e:
                logger.warning(f"xFormers not available: {e}")

            # Enable VAE slicing for lower VRAM
            self.pipe.enable_vae_slicing()
            logger.info("✅ VAE slicing enabled")

            # Enable model CPU offload if needed (optional)
            # self.pipe.enable_model_cpu_offload()

        logger.info("FLUX.1-schnell ready for generation")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 4,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate image from text prompt using FLUX.1-schnell.

        Args:
            prompt: Text description
            num_inference_steps: Number of denoising steps (1-4 recommended for schnell)
                - 1 step: Ultra-fast (~1s), lower quality
                - 4 steps: Fast (~3s), high quality (RECOMMENDED)
                - 8+ steps: Slower, marginal quality gain
            height: Output height (512 recommended)
            width: Output width (512 recommended)
            seed: Random seed for reproducibility (optional)

        Returns:
            PIL Image

        Example:
            >>> generator = FluxImageGenerator()
            >>> image = generator.generate(
            ...     "a red sports car, studio lighting",
            ...     num_inference_steps=4
            ... )
        """
        try:
            # Set seed if provided
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            else:
                generator = None

            # Generate
            result = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0,  # Schnell works best without guidance
                height=height,
                width=width,
                generator=generator
            )

            image = result.images[0]

            logger.debug(
                f"Generated {width}x{height} image in {num_inference_steps} steps"
            )

            return image

        except Exception as e:
            logger.error(f"FLUX generation failed: {e}")
            raise

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[str],
        num_inference_steps: int = 4,
        height: int = 512,
        width: int = 512
    ) -> list[Image.Image]:
        """
        Generate multiple images in a batch (more efficient).

        Args:
            prompts: List of text descriptions
            num_inference_steps: Denoising steps
            height: Output height
            width: Output width

        Returns:
            List of PIL Images
        """
        try:
            result = self.pipe(
                prompt=prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0,
                height=height,
                width=width
            )

            return result.images

        except Exception as e:
            logger.error(f"FLUX batch generation failed: {e}")
            raise

    def set_device(self, device: str):
        """Change device (e.g., for multi-GPU setups)"""
        self.device = device
        self.pipe.to(device)
        logger.info(f"Moved FLUX pipeline to {device}")

    def clear_cache(self):
        """Clear GPU cache to free VRAM"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")


# Speed benchmark reference:
# RTX 4090 (24GB):
# - 1 step: ~1.0s
# - 4 steps: ~3.0s (RECOMMENDED)
# - 8 steps: ~6.0s
#
# Quality comparison:
# - 1 step: CLIP ~0.60-0.65
# - 4 steps: CLIP ~0.70-0.75 (RECOMMENDED)
# - 8 steps: CLIP ~0.72-0.77 (marginal gain)
#
# For competitive mining: Use 4 steps (best speed/quality trade-off)
