"""
LGM Integration for Fast Direct Gaussian Generation
Use this for production: 5s generation, 100K-300K gaussians (10-30 MB files)
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import time
from loguru import logger
import io
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file

# Add LGM to path
LGM_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'LGM'))
sys.path.insert(0, LGM_PATH)

try:
    from core.models import LGM
    from core.options import Options, config_defaults
    from mvdream.pipeline_mvdream import MVDreamPipeline
    import kiui
    from kiui.op import recenter
except ImportError as e:
    logger.error(f"LGM not installed! Install from: https://github.com/3DTopia/LGM")
    logger.error(f"Error: {e}")
    raise

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class LGMGaussianGenerator:
    """
    Fast Gaussian generation using LGM

    Performance: 5 seconds per generation (guaranteed)
    Output: 100K-300K gaussians (10-30 MB files)
    """

    def __init__(self, mvdream_model: str = "ashawkey/imagedream-ipmv-diffusers", device: str = "cuda"):
        """
        Initialize LGM pipeline

        Args:
            mvdream_model: HuggingFace model path for MVDream
            device: Device to run on ('cuda' or 'cpu')
        """
        logger.info(f"Loading LGM pipeline...")
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Create options for LGM using "big" configuration
        # Big model gives maximum gaussians (128x128 splat size)
        self.opt = config_defaults['big']

        # Load LGM model
        logger.info(f"  Loading LGM model (big mode)...")
        self.model = LGM(self.opt)

        # Load pretrained weights for big model
        lgm_weights_path = os.path.join(LGM_PATH, "pretrained", "model_fp16_fixrot.safetensors")

        if not os.path.exists(lgm_weights_path):
            logger.error(f"Model weights not found at {lgm_weights_path}")
            logger.error("Please download: https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors")
            raise FileNotFoundError(f"Model weights not found: {lgm_weights_path}")

        logger.info(f"  Loading weights from {lgm_weights_path}...")
        ckpt = load_file(lgm_weights_path, device='cpu')
        self.model.load_state_dict(ckpt, strict=False)
        logger.info(f"  ✅ Loaded checkpoint")

        self.model = self.model.half().to(self.device)
        self.model.eval()

        # Prepare default rays
        self.rays_embeddings = self.model.prepare_default_rays(self.device)

        # Projection matrix for rendering
        tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        self.proj_matrix[0, 0] = 1 / tan_half_fov
        self.proj_matrix[1, 1] = 1 / tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1

        # Load MVDream pipeline for multi-view generation
        logger.info(f"  Loading MVDream pipeline for multi-view generation...")
        self.mvdream_pipe = MVDreamPipeline.from_pretrained(
            mvdream_model,  # Use mvdream_model parameter, not lgm_weights_path!
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.mvdream_pipe = self.mvdream_pipe.to(self.device)

        logger.info("✅ LGM pipeline loaded on GPU")

    async def generate_gaussian_splat(
        self,
        rgba_image: Image.Image,
        prompt: str,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 30
    ):
        """
        Generate Gaussian Splat with LGM

        Args:
            rgba_image: PIL Image (RGBA) from FLUX → background removal
            prompt: Text prompt (for logging/context, not used by LGM)
            guidance_scale: CFG strength (5.0 is good default)
            num_inference_steps: Diffusion steps (30 recommended)

        Returns:
            ply_bytes: Binary PLY data
            gs_model: GaussianModel for validation
            timings: Dict of timing info
        """
        start_time = time.time()
        logger.info(f"  [3/4] Generating Gaussian Splat with LGM...")

        try:
            # Convert PIL Image to numpy array
            image_np = np.array(rgba_image)  # [H, W, 4]

            # Recenter the image (ensures object is centered)
            t1 = time.time()
            mask = image_np[..., -1] > 0
            image_np = recenter(image_np, mask, border_ratio=0.2)
            logger.debug(f"    Recentering done ({time.time()-t1:.2f}s)")

            # Convert to float32 [0, 1]
            image_np = image_np.astype(np.float32) / 255.0

            # RGBA to RGB with white background
            if image_np.shape[-1] == 4:
                image_np = image_np[..., :3] * image_np[..., 3:4] + (1 - image_np[..., 3:4])

            # Generate multi-view images using MVDream
            t2 = time.time()
            logger.info(f"    Generating multi-view images...")
            mv_image = self.mvdream_pipe(
                '',  # Empty prompt for MVDream (it uses the image)
                image_np,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                elevation=0
            )

            # Reorder multi-view images: [front, side, back, side] → [side, back, side, front]
            mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0)  # [4, 256, 256, 3]
            logger.debug(f"    Multi-view generation done ({time.time()-t2:.2f}s)")

            # Prepare input for LGM model
            t3 = time.time()
            input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(self.device)  # [4, 3, 256, 256]
            input_image = F.interpolate(input_image, size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False)
            input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            input_image = torch.cat([input_image, self.rays_embeddings], dim=1).unsqueeze(0)  # [1, 4, 9, H, W]

            # Generate gaussians with LGM
            logger.info(f"    Running LGM forward pass...")
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    gaussians = self.model.forward_gaussians(input_image)

            logger.debug(f"    LGM forward pass done ({time.time()-t3:.2f}s)")

            # Save gaussians to PLY bytes
            t4 = time.time()
            ply_bytes = self._gaussians_to_ply_bytes(gaussians)

            # Get statistics
            num_gaussians = self._count_gaussians_in_ply(ply_bytes)
            file_size_mb = len(ply_bytes) / (1024 * 1024)

            elapsed = time.time() - start_time
            logger.info(f"  ✅ LGM generation done ({elapsed:.2f}s)")
            logger.info(f"     Generated {num_gaussians:,} gaussians ({file_size_mb:.1f} MB)")

            timings = {
                "lgm": elapsed,
                "total_3d": elapsed
            }

            # Create GaussianModel for rendering validation
            gs_model = self._create_gaussian_model_from_ply(ply_bytes)

            return ply_bytes, gs_model, timings

        except Exception as e:
            logger.error(f"LGM generation failed: {e}", exc_info=True)
            raise

    def _gaussians_to_ply_bytes(self, gaussians):
        """Convert LGM gaussians to PLY bytes"""
        import tempfile
        import os

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Use LGM's built-in PLY saver
            self.model.gs.save_ply(gaussians, tmp_path)

            # Read PLY bytes
            with open(tmp_path, 'rb') as f:
                ply_bytes = f.read()

            return ply_bytes

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _count_gaussians_in_ply(self, ply_bytes: bytes) -> int:
        """Parse PLY header to count gaussians"""
        try:
            header_end = ply_bytes.find(b"end_header\n")
            if header_end == -1:
                return 0

            header = ply_bytes[:header_end].decode('utf-8')
            for line in header.split('\n'):
                if line.startswith('element vertex'):
                    return int(line.split()[-1])
            return 0
        except:
            return 0

    def _create_gaussian_model_from_ply(self, ply_bytes: bytes):
        """Create GaussianModel from PLY for rendering validation"""
        try:
            from DreamGaussianLib.GaussianSplattingModel import GaussianModel
            import tempfile
            import os

            # Save PLY to temp file
            with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp:
                tmp.write(ply_bytes)
                tmp_path = tmp.name

            # Load into GaussianModel
            gs_model = GaussianModel(sh_degree=2)  # LGM uses SH degree 2
            gs_model.load_ply(tmp_path)

            # Clean up
            os.unlink(tmp_path)

            logger.debug(f"Created GaussianModel from PLY for rendering")
            return gs_model

        except Exception as e:
            logger.warning(f"Failed to create GaussianModel from PLY: {e}")
            return None


async def generate_with_lgm(
    rgba_image: Image.Image,
    prompt: str,
    lgm_generator: LGMGaussianGenerator,
    guidance_scale: float = 5.0
):
    """
    Wrapper function matching expected interface

    Drop-in replacement for InstantMesh in serve_competitive.py
    """
    return await lgm_generator.generate_gaussian_splat(
        rgba_image=rgba_image,
        prompt=prompt,
        guidance_scale=guidance_scale
    )
