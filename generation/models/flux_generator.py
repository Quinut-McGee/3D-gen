"""
Stable Diffusion 1.5 integration for fast, reliable text-to-image generation.

SD 1.5 is battle-tested for competitive mining:
- 20-25 steps for generation (fast and stable)
- Good quality (CLIP 0.60-0.68)
- Free for commercial use
- Very stable, ~4GB VRAM
- Most widely deployed SD model
"""

from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from typing import Optional, Union
from loguru import logger


class FluxImageGenerator:  # Keep class name for compatibility
    """
    Stable Diffusion 1.5 text-to-image generator.

    This replaces MVDream for the initial image generation step.
    Battle-tested, fast, and reliable for production mining.
    """

    def __init__(
        self,
        device: str = "cuda",
        torch_dtype=torch.float16,  # fp16 is standard for SD
        enable_optimization: bool = True
    ):
        """
        Initialize SDXL-Turbo pipeline with lazy loading.

        Args:
            device: CUDA device
            torch_dtype: Data type (fp16 recommended for SDXL)
            enable_optimization: Enable memory and speed optimizations
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self.enable_optimization = enable_optimization
        self.pipe = None
        self.is_loaded = False

        logger.info("Stable Diffusion 1.5 configured for lazy loading (will load on first generation)")
        logger.info("✅ SD 1.5 will be loaded on-demand to save VRAM")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 20,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate image from text prompt using Stable Diffusion 1.5.

        Args:
            prompt: Text description
            num_inference_steps: Number of denoising steps (15-25 recommended)
                - 15 steps: Fast (~3s), acceptable quality
                - 20 steps: Balanced (~4s), good quality (RECOMMENDED)
                - 25 steps: Slower (~5s), better quality
            height: Output height (512 recommended)
            width: Output width (512 recommended)
            seed: Random seed for reproducibility (optional)

        Returns:
            PIL Image

        Example:
            >>> generator = FluxImageGenerator()
            >>> image = generator.generate(
            ...     "a red sports car, studio lighting",
            ...     num_inference_steps=20
            ... )
        """
        try:
            # Ensure pipeline is loaded
            if not self.is_loaded or self.pipe is None:
                logger.warning("Pipeline not loaded, loading now...")
                self._load_pipeline()

            # Set seed if provided
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            else:
                generator = None

            # CUDA sync before generation to ensure clean state
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Generate (SD 1.5 uses standard guidance_scale)
            result = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,  # Standard SD guidance
                height=height,
                width=width,
                generator=generator
            )

            # CUDA sync after generation
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            image = result.images[0]

            logger.debug(
                f"Generated {width}x{height} image in {num_inference_steps} steps"
            )

            # Clean up GPU cache immediately after generation
            self.clear_cache()

            return image

        except Exception as e:
            logger.error(f"SD 1.5 generation failed: {e}", exc_info=True)
            # Clean up on error too
            self.clear_cache()
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

            # Clean up GPU cache immediately after generation
            self.clear_cache()

            return result.images

        except Exception as e:
            logger.error(f"FLUX batch generation failed: {e}")
            # Clean up on error too
            self.clear_cache()
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
            torch.cuda.synchronize()
            logger.debug("Cleared CUDA cache")

    def offload_to_cpu(self):
        """Offload model to CPU to free GPU memory, or unload entirely"""
        if self.is_loaded and self.pipe is not None:
            # Instead of moving to CPU, completely unload to free all memory
            logger.debug("Unloading FLUX to free GPU memory...")
            del self.pipe
            self.pipe = None
            self.is_loaded = False

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.debug("✅ FLUX unloaded and memory freed")

    def _load_pipeline(self):
        """Lazy load Stable Diffusion 1.5 pipeline - battle-tested and fast"""
        if not self.is_loaded:
            logger.info("Loading Stable Diffusion 1.5...")

            # Load SD 1.5 pipeline - most stable SD model
            # Use local_files_only to avoid network issues
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=self.torch_dtype,
                safety_checker=None,  # Disable for speed
                local_files_only=True  # Force use of cached model
            )

            # Move to GPU
            logger.debug("Moving pipeline to GPU...")
            self.pipe.to(self.device)

            if self.enable_optimization:
                # DISABLE xFormers for now - can cause CUDA conflicts
                # try:
                #     self.pipe.enable_xformers_memory_efficient_attention()
                #     logger.info("✅ xFormers enabled")
                # except Exception as e:
                #     logger.warning(f"xFormers not available: {e}")
                logger.info("⚠️  xFormers disabled to prevent CUDA conflicts")

                # Enable VAE slicing to reduce memory
                try:
                    self.pipe.enable_vae_slicing()
                    logger.info("✅ VAE slicing enabled")
                except Exception as e:
                    logger.warning(f"VAE slicing not available: {e}")

                # Enable attention slicing for lower memory
                try:
                    self.pipe.enable_attention_slicing(1)
                    logger.info("✅ Attention slicing enabled")
                except Exception as e:
                    logger.warning(f"Attention slicing not available: {e}")

            self.is_loaded = True
            logger.info("✅ Stable Diffusion 1.5 ready (~4GB VRAM, 3-5s generation)")

    def ensure_on_gpu(self):
        """Ensure model is loaded and ready on GPU"""
        if not self.is_loaded:
            self._load_pipeline()
        elif self.pipe is not None:
            self.pipe.to(self.device)
            logger.debug("Moved FLUX to GPU")


# Speed benchmark reference (Stable Diffusion 1.5):
# RTX 4090 (24GB):
# - 15 steps: ~3s
# - 20 steps: ~4s (RECOMMENDED)
# - 25 steps: ~5s
#
# Quality comparison:
# - 15 steps: CLIP ~0.58-0.65
# - 20 steps: CLIP ~0.60-0.68 (RECOMMENDED)
# - 25 steps: CLIP ~0.62-0.69
#
# Memory usage: ~4GB VRAM
# For competitive mining: Use 20 steps (best speed/quality trade-off)
