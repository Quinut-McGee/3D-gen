"""
SDXL-Turbo integration for ultra-fast, high-quality text-to-image generation.

SDXL-Turbo is perfect for competitive mining:
- 1-4 steps for generation (vs 50+ for diffusion models)
- High quality based on SDXL
- Free for commercial use
- No gating/authentication required
- Smaller download (~7GB vs 12GB for FLUX)
"""

from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
from typing import Optional
from loguru import logger


class SDXLTurboGenerator:
    """
    SDXL-Turbo text-to-image generator.

    Alternative to FLUX that doesn't require HuggingFace authentication
    and uses less disk space.
    """

    def __init__(
        self,
        device: str = "cuda",
        torch_dtype=torch.float16,
        enable_optimization: bool = True
    ):
        """
        Initialize SDXL-Turbo pipeline.

        Args:
            device: CUDA device
            torch_dtype: Data type (float16 recommended for speed)
            enable_optimization: Enable memory and speed optimizations
        """
        self.device = device
        self.torch_dtype = torch_dtype

        logger.info("Loading SDXL-Turbo pipeline...")

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch_dtype,
            variant="fp16"
        )

        self.pipe.to(device)

        # Optimizations
        if enable_optimization:
            logger.info("Applying optimizations...")

            # Enable memory efficient attention
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("✅ xFormers enabled")
            except Exception as e:
                logger.warning(f"xFormers not available: {e}")

            # Enable VAE slicing for lower VRAM
            self.pipe.enable_vae_slicing()
            logger.info("✅ VAE slicing enabled")

        logger.info("SDXL-Turbo ready for generation")

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
        Generate image from text prompt using SDXL-Turbo.

        Args:
            prompt: Text description
            num_inference_steps: Number of denoising steps (1-4 recommended)
                - 1 step: Ultra-fast (~1s), lower quality
                - 4 steps: Fast (~2-3s), high quality (RECOMMENDED)
            height: Output height (512 recommended)
            width: Output width (512 recommended)
            seed: Random seed for reproducibility (optional)

        Returns:
            PIL Image
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
                guidance_scale=0.0,  # Turbo works best without guidance
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
            logger.error(f"SDXL-Turbo generation failed: {e}")
            raise

    def clear_cache(self):
        """Clear GPU cache to free VRAM"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")


# Speed benchmark reference (RTX 4090):
# - 1 step: ~1.0s
# - 2 steps: ~1.5s
# - 4 steps: ~2.5s (RECOMMENDED)
#
# Quality comparison:
# - 1 step: CLIP ~0.55-0.60
# - 2 steps: CLIP ~0.65-0.70
# - 4 steps: CLIP ~0.70-0.75 (RECOMMENDED)
#
# For competitive mining: Use 4 steps
