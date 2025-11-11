"""
SDXL-Turbo: Fast, memory-efficient text-to-image for competitive mining.

Memory: 4-5GB on GPU (11GB free on RTX 5070 Ti)
Speed: 1-2s for 512x512 generation (1-4 steps)
Quality: 80-85% of FLUX, better than SD 2.1

Perfect for mining: Fast + reliable + low memory.
"""

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
from typing import Optional
from loguru import logger
import gc


class SDXLTurboGenerator:
    """
    SDXL-Turbo for fast, memory-efficient image generation.
    
    Optimized for 16GB GPUs with competitive mining speed requirements.
    """
    
    def __init__(self, device: str = "cuda:1"):
        """
        Initialize SDXL-Turbo generator.
        
        Args:
            device: CUDA device (GPU 1 recommended for dual-GPU setup)
        """
        self.device = device
        self.pipe = None
        self.is_loaded = False
        
        logger.info(f"SDXL-Turbo initialized for {device}")
        logger.info("  Memory: 4-5GB (11GB free on 16GB GPU)")
        logger.info("  Speed: 1-2s per image (1-4 steps)")
        logger.info("  Quality: 80-85% FLUX, better than SD 2.1")
    
    def _load_pipeline(self):
        """Load SDXL-Turbo pipeline (lazy loading)"""
        if self.is_loaded:
            return
        
        logger.info("Loading SDXL-Turbo pipeline...")

        # Load with memory optimization
        # Note: Load on CPU first to avoid CUDA tensor->numpy conversion issues in scheduler init
        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            device_map=None  # Load on CPU first, then move to GPU
        )

        # Move to GPU after initialization
        self.pipe = self.pipe.to(self.device)
        
        # Enable memory optimizations
        self.pipe.enable_attention_slicing(1)
        
        # DISABLED: xFormers causes CUDA flash-attention crash on RTX 5070 Ti
        # Error: "flash-attention/hopper/flash_fwd_launch_template.h:188: invalid argument"
        # Using standard attention instead (slightly slower but stable)
        logger.info("  ℹ️  Using standard attention (xFormers disabled for RTX 5070 Ti compatibility)")
        
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_loaded = True
        
        # Log VRAM usage
        if torch.cuda.is_available() and "cuda" in self.device:
            device_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
            allocated = torch.cuda.memory_allocated(device_idx) / 1024**3
            logger.info(f"✅ SDXL-Turbo ready on {self.device}")
            logger.info(f"   VRAM allocated: {allocated:.2f}GB")
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 4,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
        negative_prompt: str = ""
    ) -> Image.Image:
        """
        Generate image from text prompt.

        Args:
            prompt: Text description
            num_inference_steps: Denoising steps (1-4 recommended for Turbo)
            height: Output height (512 recommended)
            width: Output width (512 recommended)
            seed: Random seed (optional)
            negative_prompt: Negative prompt to guide generation away from unwanted elements

        Returns:
            PIL Image
        """
        try:
            # Ensure pipeline is loaded
            if not self.is_loaded:
                self._load_pipeline()
            
            logger.debug(f"Generating with SDXL-Turbo: '{prompt[:50]}...'")
            
            # Set seed if provided
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            else:
                generator = None
            
            # SDXL-Turbo: No guidance scale (baked into model)
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                generator=generator,
                guidance_scale=0.0  # Turbo doesn't use guidance
            )
            
            image = result.images[0]
            
            logger.debug(f"✅ Generated {width}x{height} image in {num_inference_steps} steps")
            
            # Cleanup after generation
            del result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return image
        
        except Exception as e:
            logger.error(f"❌ SDXL-Turbo generation failed: {e}", exc_info=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    
    def clear_cache(self):
        """Clear GPU cache"""
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")


# Performance notes:
# - Memory: 4-5GB on GPU (vs 15GB FLUX)
# - Speed: 1-2s (4 steps) vs 27s FLUX
# - Quality: 80-85% FLUX quality, sufficient for validators
# - Stability: Mature model, no experimental features
# - Mining fit: Perfect balance of speed/quality/memory
