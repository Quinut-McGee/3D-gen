"""
TRELLIS Microservice for 3D Gaussian Generation
Runs in trellis-env with PyTorch 2.5 + Kaolin
Port: 10008 (isolated from main miner on port 10006)

Performance: 5s generation, 256K gaussians, 16.6 MB files
"""

import os
os.environ['SPCONV_ALGO'] = 'native'  # Critical for performance

import sys
# Add TRELLIS to Python path (needed for local package import)
sys.path.insert(0, '/home/kobe/404-gen/v1/3D-gen/TRELLIS')

import time
import base64
import tempfile
import gc
from io import BytesIO
from typing import Dict, Optional
import argparse
import traceback

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import uvicorn
from loguru import logger

# Custom PLY writer to prevent TRELLIS save_ply() corruption
from clean_ply_writer import save_clean_ply

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("logs/trellis_microservice.log", rotation="500 MB", retention="10 days", level="DEBUG")

app = FastAPI(title="TRELLIS Microservice", version="1.0.0")

# Global pipeline - lazy loaded on demand (Option B: unload after generation)
pipeline = None
PIPELINE_LOADED = False  # NOT USED - we lazy load now


class GenerateRequest(BaseModel):
    """Request model for gaussian generation"""
    image_base64: str
    seed: int = 42
    timeout: int = 30  # seconds


class GenerateResponse(BaseModel):
    """Response model with PLY and statistics"""
    ply_base64: str
    num_gaussians: int
    file_size_mb: float
    generation_time: float
    success: bool
    error: Optional[str] = None


@app.on_event("startup")
async def startup():
    """Startup - PRE-LOAD pipeline (FLUX uses CPU offload, so we have VRAM!)"""
    global pipeline, PIPELINE_LOADED

    logger.info("=" * 70)
    logger.info("🚀 TRELLIS MICROSERVICE STARTING (PRE-LOADING MODE)")
    logger.info("=" * 70)
    logger.info("⚡ FLUX uses CPU offload (~2-3GB), leaving room for TRELLIS (~13GB)")
    logger.info("⚡ Pre-loading TRELLIS to avoid 24s lazy-load delay...")

    # Pre-load pipeline
    from trellis.pipelines import TrellisImageTo3DPipeline
    import torch
    import time

    load_start = time.time()
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    load_time = time.time() - load_start
    PIPELINE_LOADED = True

    logger.info(f"✅ TRELLIS pre-loaded in {load_time:.1f}s")
    logger.info("📡 Ready for FAST generation (~5-6s per request)")
    logger.info("=" * 70)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if PIPELINE_LOADED else "loading",
        "pipeline_loaded": PIPELINE_LOADED,
        "service": "trellis-microservice",
        "version": "1.0.0"
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_gaussian(request: GenerateRequest) -> GenerateResponse:
    """
    Generate 3D Gaussian Splat from RGBA image (LAZY LOADING MODE)

    Args:
        request: Contains base64-encoded RGBA image and generation params

    Returns:
        GenerateResponse with base64-encoded PLY and statistics
    """
    global pipeline
    start_time = time.time()

    try:
        # Pipeline is pre-loaded at startup - ready to use!
        # Decode input image
        logger.info("📥 Received generation request")
        image_bytes = base64.b64decode(request.image_base64)
        rgba_image = Image.open(BytesIO(image_bytes)).convert('RGB')
        logger.debug(f"   Image size: {rgba_image.size}, mode: {rgba_image.mode}")

        # Generate with TRELLIS (HIGH-QUALITY MODE)
        logger.info("🎨 Generating 3D gaussians with TRELLIS (HIGH-QUALITY)...")
        gen_start = time.time()

        outputs = pipeline.run(
            rgba_image,
            seed=request.seed,
            # OPTIMIZED PARAMETERS for dense voxel generation + quality
            # Sparse structure: Detects voxels on object surface
            # SLAT: Fills voxels with gaussian details
            #
            # Post-Mortem Optimization (Nov 5, 2025):
            # Increased sampling steps to improve baseline quality across ALL validators
            # Trade-off: +3-4s generation time, +50-100K gaussians, +10-15% success rate
            # CFG kept at proven values (9.0/4.0) - earlier testing showed these work best
            #
            # Phase 5 Optimization (Nov 6, 2025):
            # Further increased sampling to boost gaussian density for validator acceptance
            # Combined with bbox normalization fix for comprehensive rejection prevention
            # Expected: 398K → 500-600K gaussians, +15-20% success rate
            sparse_structure_sampler_params={
                "steps": 80,  # Phase 5: Increased from 60 (+33% for denser geometry)
                "cfg_strength": 9.0,  # Phase 3 optimal value (tested better than 5.0)
            },
            slat_sampler_params={
                "steps": 60,  # Phase 5: Increased from 50 (+20% for finer detail)
                "cfg_strength": 4.0,  # Phase 3 optimal value (balanced quality)
            },
        )

        gen_time = time.time() - gen_start
        logger.info(f"   Generation completed in {gen_time:.2f}s")

        # Extract gaussian output
        gaussian_output = outputs['gaussian'][0]

        # 🔬 PHASE 3 DIAGNOSTIC: Log raw TRELLIS opacity BEFORE normalization
        try:
            import torch
            import numpy as np
            if hasattr(gaussian_output, '_opacity'):
                raw_opacities = gaussian_output._opacity.detach().cpu().numpy().flatten()
                logger.info(f"🔬 RAW TRELLIS OPACITY (before normalization):")
                logger.info(f"   mean: {raw_opacities.mean():.4f}, std: {raw_opacities.std():.4f}")
                logger.info(f"   min: {raw_opacities.min():.4f}, max: {raw_opacities.max():.4f}")
                logger.info(f"   num_inf: {np.isinf(raw_opacities).sum()}, num_nan: {np.isnan(raw_opacities).sum()}")
            else:
                logger.warning(f"   No _opacity attribute found in gaussian_output")
        except Exception as e:
            logger.warning(f"   Could not inspect raw TRELLIS opacities: {e}")

        # 🔧 PHASE 3 FIX: Pre-save opacity normalization to prevent corruption
        # Diagnostic analysis showed TRELLIS generates extreme outliers (±15) that corrupt save_ply()
        # This 3-step normalization prevents corruption while preserving relative opacity structure
        try:
            import torch
            if hasattr(gaussian_output, '_opacity'):
                opacities = gaussian_output._opacity

                # Get current statistics
                opacity_mean = opacities.mean().item()
                opacity_std = opacities.std().item()
                opacity_min = opacities.min().item()
                opacity_max = opacities.max().item()

                # Check if normalization is needed
                needs_fix = (
                    opacity_mean < 7.0 or  # Unhealthy mean (raised from 4.0 to account for save_ply drop)
                    opacity_min < -10.0 or  # Extreme negative outlier
                    opacity_max > 15.0 or   # Extreme positive outlier
                    torch.isinf(opacities).any() or
                    torch.isnan(opacities).any()
                )

                if needs_fix:
                    logger.warning(f"🔧 Opacity corruption risk detected, applying normalization")
                    logger.warning(f"   Before: mean={opacity_mean:.3f}, range=[{opacity_min:.3f}, {opacity_max:.3f}]")

                    # STEP 1: Clamp extreme outliers (prevents save_ply inf/nan corruption)
                    opacities_clamped = torch.clamp(opacities, -9.21, 12.15)  # TRELLIS natural range from analysis

                    # STEP 2: Replace inf/nan with median (safety check)
                    if torch.isinf(opacities_clamped).any() or torch.isnan(opacities_clamped).any():
                        valid_mask = torch.isfinite(opacities_clamped)
                        if valid_mask.any():
                            median_opacity = torch.median(opacities_clamped[valid_mask])
                        else:
                            median_opacity = torch.tensor(6.5)  # Fallback to healthy value
                        opacities_clamped[~valid_mask] = median_opacity

                    # STEP 3: Smart Adaptive Normalization - Respect TRELLIS's Natural Healthy Range
                    #
                    # TRELLIS naturally generates opacity in different ranges:
                    # - Healthy: 2.0-6.0 (gives 88%-98% actual opacity after sigmoid)
                    # - Broken: < 2.0 (too transparent) or > 6.0 (too opaque)
                    #
                    # Strategy: Only normalize if OUTSIDE healthy range, otherwise preserve TRELLIS output
                    current_mean = opacities_clamped.mean().item()

                    # Define TRELLIS's natural healthy range (empirically determined)
                    HEALTHY_MIN = 2.0   # Below this = too transparent (invisible gaussians)
                    HEALTHY_MAX = 6.0   # Above this = too opaque (solid blob, no blending)
                    TARGET_OPTIMAL = 4.0  # Sweet spot (gives ~98% opacity - natural TRELLIS output)

                    logger.info(f"🔬 SMART NORMALIZATION:")
                    logger.info(f"   Current opacity mean: {current_mean:.3f}")
                    logger.info(f"   Healthy range: [{HEALTHY_MIN}, {HEALTHY_MAX}]")

                    if current_mean < HEALTHY_MIN:
                        # TOO TRANSPARENT: Raise to optimal
                        target_mean = TARGET_OPTIMAL
                        logger.warning(f"   ⚠️ TOO TRANSPARENT (mean={current_mean:.3f} < {HEALTHY_MIN})")
                        logger.info(f"   Normalizing to target: {target_mean:.1f}")
                    elif current_mean > HEALTHY_MAX:
                        # TOO OPAQUE: Lower to optimal
                        target_mean = TARGET_OPTIMAL
                        logger.warning(f"   ⚠️ TOO OPAQUE (mean={current_mean:.3f} > {HEALTHY_MAX})")
                        logger.info(f"   Normalizing to target: {target_mean:.1f}")
                    else:
                        # ALREADY HEALTHY: Don't touch! TRELLIS knows what it's doing
                        target_mean = None
                        logger.info(f"   ✅ Opacity HEALTHY (in range [{HEALTHY_MIN}, {HEALTHY_MAX}]) - preserving TRELLIS output")

                    if target_mean is not None:
                        # Apply normalization shift
                        shift_amount = target_mean - current_mean
                        opacities_shifted = opacities_clamped + shift_amount
                        opacities_fixed = torch.clamp(opacities_shifted, -9.21, 12.15)
                        logger.info(f"   Applied shift: {shift_amount:+.3f} (final mean: {opacities_fixed.mean().item():.3f})")
                    else:
                        # Keep TRELLIS's natural output
                        opacities_fixed = opacities_clamped

                    # Update the gaussian output
                    gaussian_output._opacity = opacities_fixed

                    # Log results
                    new_mean = opacities_fixed.mean().item()
                    new_min = opacities_fixed.min().item()
                    new_max = opacities_fixed.max().item()
                    logger.info(f"   After: mean={new_mean:.3f}, range=[{new_min:.3f}, {new_max:.3f}]")
                else:
                    logger.debug(f"   Opacity healthy, no normalization needed (mean={opacity_mean:.3f})")

        except Exception as e:
            logger.error(f"🔧 Opacity normalization failed: {e}")
            # Continue anyway - downstream ply_fixer.py will catch any remaining issues

        # Save to PLY using CLEAN writer (bypasses TRELLIS corruption)
        logger.info("💾 Saving PLY file with clean writer...")
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Use our custom writer instead of gaussian_output.save_ply()
            # This prevents the inverse_sigmoid/log corruption that was causing rejections
            save_clean_ply(gaussian_output, tmp_path)

            # Read PLY bytes
            with open(tmp_path, 'rb') as f:
                ply_bytes = f.read()

            # Count gaussians from PLY header
            header_end = ply_bytes.find(b"end_header\n")
            header = ply_bytes[:header_end].decode('utf-8')
            num_gaussians = 0
            for line in header.split('\n'):
                if line.startswith('element vertex'):
                    num_gaussians = int(line.split()[-1])
                    break

            file_size_mb = len(ply_bytes) / (1024 * 1024)

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # Encode PLY to base64
        ply_base64 = base64.b64encode(ply_bytes).decode('utf-8')

        total_time = time.time() - start_time

        logger.info("✅ Generation successful!")
        logger.info(f"   Gaussians: {num_gaussians:,}")
        logger.info(f"   File size: {file_size_mb:.1f} MB")
        logger.info(f"   Total time: {total_time:.2f}s")

        # KEEP PIPELINE LOADED: With FLUX using CPU offload (~2-3GB), we have room!
        # FLUX CPU offload: ~2-3GB
        # TRELLIS loaded: ~13GB
        # Total: ~15-16GB (fits in 24GB GPU!)
        logger.info("✅ TRELLIS staying loaded (FLUX uses CPU offload, no VRAM conflict)")

        # MEMORY LEAK FIX: Explicitly delete GPU tensor objects before cache clear
        # These objects contain large GPU tensors (128K-512K gaussians) that accumulate
        # over time if not explicitly freed. Python GC is non-deterministic and may not
        # run for 90+ minutes, causing VRAM to fill up and operations to slow down.
        import torch

        # Delete objects holding GPU tensors
        del gaussian_output  # Gaussian splat object with GPU tensors
        del outputs          # Dict containing intermediate GPU tensors
        del rgba_image       # Input image (CPU, but still good practice)

        # Force immediate garbage collection to free GPU memory
        gc.collect()

        # Now clear CUDA cache to reclaim freed memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("   Freed GPU tensors and cleared CUDA cache (TRELLIS pipeline still loaded)")

        return GenerateResponse(
            ply_base64=ply_base64,
            num_gaussians=num_gaussians,
            file_size_mb=file_size_mb,
            generation_time=gen_time,
            success=True,
            error=None
        )

    except Exception as e:
        error_msg = f"Generation failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        logger.error(traceback.format_exc())

        return GenerateResponse(
            ply_base64="",
            num_gaussians=0,
            file_size_mb=0.0,
            generation_time=time.time() - start_time,
            success=False,
            error=error_msg
        )


@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "TRELLIS Microservice",
        "version": "1.0.0",
        "status": "running" if PIPELINE_LOADED else "loading",
        "port": 10008,
        "endpoints": {
            "health": "/health",
            "generate": "/generate (POST)",
            "docs": "/docs"
        },
        "performance": {
            "generation_time": "~5s",
            "gaussian_count": "~256K",
            "file_size": "~16.6 MB"
        }
    }


def main():
    parser = argparse.ArgumentParser(description="TRELLIS Microservice")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=10008, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (must be 1 for GPU)")
    args = parser.parse_args()

    logger.info(f"Starting TRELLIS microservice on {args.host}:{args.port}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )


if __name__ == "__main__":
    main()
