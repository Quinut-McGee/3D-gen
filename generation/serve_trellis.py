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
            sparse_structure_sampler_params={
                "steps": 60,  # TEST 3: Maximum refinement
                "cfg_strength": 5.0,  # TEST 3: Much lower guidance to eliminate blobs
            },
            slat_sampler_params={
                "steps": 50,  # TEST 3: Maximum SLAT refinement
                "cfg_strength": 2.5,  # TEST 3: Minimal guidance for realistic scales
            },
        )

        gen_time = time.time() - gen_start
        logger.info(f"   Generation completed in {gen_time:.2f}s")

        # Extract gaussian output
        gaussian_output = outputs['gaussian'][0]

        # Save to PLY
        logger.info("💾 Saving PLY file...")
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            gaussian_output.save_ply(tmp_path)

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
