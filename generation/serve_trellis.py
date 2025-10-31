"""
TRELLIS Microservice for 3D Gaussian Generation
Runs in trellis-env with PyTorch 2.5 + Kaolin
Port: 10008 (isolated from main miner on port 10006)

Performance: 5s generation, 256K gaussians, 16.6 MB files
"""

import os
os.environ['SPCONV_ALGO'] = 'native'  # Critical for performance

import sys
import time
import base64
import tempfile
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

# Global pipeline - loaded once at startup
pipeline = None
PIPELINE_LOADED = False


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
async def load_pipeline():
    """Load TRELLIS pipeline on startup (takes ~27s)"""
    global pipeline, PIPELINE_LOADED

    logger.info("=" * 70)
    logger.info("🚀 TRELLIS MICROSERVICE STARTING")
    logger.info("=" * 70)

    try:
        logger.info("Loading TRELLIS-image-large pipeline...")
        start_time = time.time()

        from trellis.pipelines import TrellisImageTo3DPipeline

        pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        pipeline.cuda()

        load_time = time.time() - start_time
        PIPELINE_LOADED = True

        logger.info(f"✅ TRELLIS pipeline loaded in {load_time:.1f}s")
        logger.info("📡 Ready to accept requests on port 10008")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"❌ Failed to load TRELLIS pipeline: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


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
    Generate 3D Gaussian Splat from RGBA image

    Args:
        request: Contains base64-encoded RGBA image and generation params

    Returns:
        GenerateResponse with base64-encoded PLY and statistics
    """
    if not PIPELINE_LOADED:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet")

    start_time = time.time()

    try:
        # Decode input image
        logger.info("📥 Received generation request")
        image_bytes = base64.b64decode(request.image_base64)
        rgba_image = Image.open(BytesIO(image_bytes)).convert('RGB')
        logger.debug(f"   Image size: {rgba_image.size}, mode: {rgba_image.mode}")

        # Generate with TRELLIS
        logger.info("🎨 Generating 3D gaussians with TRELLIS...")
        gen_start = time.time()

        outputs = pipeline.run(
            rgba_image,
            seed=request.seed,
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
