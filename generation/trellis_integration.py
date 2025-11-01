"""
TRELLIS Integration for Native Gaussian Generation
Direct gaussian splat generation - no mesh intermediate step!

Performance: 5s generation, 256K gaussians, 16.6 MB files
"""

import io
import base64
import httpx
import time
import tempfile
import os
from loguru import logger
from PIL import Image


async def generate_with_trellis(rgba_image, prompt, trellis_url="http://localhost:10008"):
    """
    Generate 3D Gaussian Splat using TRELLIS microservice.

    Args:
        rgba_image: PIL Image (RGBA) from FLUX → background removal
        prompt: Text prompt for logging/debugging (not used by TRELLIS)
        trellis_url: TRELLIS microservice URL

    Returns:
        ply_bytes: Binary PLY data
        gs_model: GaussianModel for validation (loaded from PLY)
        timings: Dict of timing info
    """

    # Step 3: Direct Gaussian generation with TRELLIS microservice (5s)
    t3_start = time.time()
    logger.info("  [3/4] Generating 3D Gaussians with TRELLIS microservice...")

    try:
        # Convert RGBA image to RGB (TRELLIS expects RGB)
        if rgba_image.mode == 'RGBA':
            rgb_image = rgba_image.convert('RGB')
        else:
            rgb_image = rgba_image

        # Convert image to base64 for HTTP transfer
        buffer = io.BytesIO()
        rgb_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # AGGRESSIVE CLEANUP: Free GPU memory before calling TRELLIS
        import torch
        import gc
        logger.debug("🧹 Clearing GPU cache before TRELLIS...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

        # Call TRELLIS microservice
        logger.debug(f"  Calling TRELLIS at {trellis_url}/generate...")
        async with httpx.AsyncClient(timeout=60.0) as client:  # Longer timeout for TRELLIS (24s load + 5s gen)
            response = await client.post(
                f"{trellis_url}/generate",
                json={
                    "image_base64": image_base64,
                    "seed": 42,  # Fixed seed for reproducibility
                    "timeout": 30
                }
            )

            if response.status_code != 200:
                error_detail = response.json().get('detail', 'Unknown error')
                raise RuntimeError(f"TRELLIS service returned status {response.status_code}: {error_detail}")

            result = response.json()

        # Check if generation succeeded
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error")
            raise RuntimeError(f"TRELLIS generation failed: {error_msg}")

        # Decode PLY from base64
        ply_bytes = base64.b64decode(result["ply_base64"])

        t3_end = time.time()
        logger.info(f"  ✅ TRELLIS generation done ({t3_end-t3_start:.2f}s, service: {result['generation_time']:.2f}s)")
        logger.info(f"     Gaussians: {result['num_gaussians']:,}, File size: {result['file_size_mb']:.1f} MB")

    except httpx.TimeoutException:
        logger.error("TRELLIS microservice timeout (60s)")
        raise RuntimeError("TRELLIS microservice timeout")
    except Exception as e:
        logger.error(f"TRELLIS microservice call failed: {e}", exc_info=True)
        raise

    # Step 4: Create GaussianModel from PLY for validation (0.1s)
    t4_start = time.time()
    logger.info("  [4/4] Loading Gaussian model for validation...")

    try:
        # Import GaussianModel as a package (handles relative imports correctly)
        import sys
        generation_path = '/home/kobe/404-gen/v1/3D-gen/generation'
        if generation_path not in sys.path:
            sys.path.insert(0, generation_path)
        from DreamGaussianLib.GaussianSplattingModel import GaussianModel

        # Save PLY to temp file
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp:
            tmp.write(ply_bytes)
            tmp_path = tmp.name

        # Load into GaussianModel
        gs_model = GaussianModel(sh_degree=2)  # TRELLIS uses SH degree 2
        gs_model.load_ply(tmp_path)

        # Clean up temp file
        os.unlink(tmp_path)

        t4_end = time.time()
        logger.info(f"  ✅ Gaussian model loaded ({t4_end-t4_start:.2f}s)")

    except Exception as e:
        logger.warning(f"Failed to load GaussianModel for validation: {e}")
        logger.warning("Continuing without validation model (PLY is still valid)")
        gs_model = None
        t4_end = time.time()

    timings = {
        "trellis": t3_end - t3_start,
        "model_load": t4_end - t4_start if gs_model else 0.0,
        "total_3d": t4_end - t3_start
    }

    return ply_bytes, gs_model, timings
