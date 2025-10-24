"""
404-GEN COMPETITIVE GENERATION SERVICE

This is the production-ready competitive miner pipeline:
1. Stable Diffusion 1.5: Fast text-to-image (20 steps = 4s)
2. rembg: Background removal
3. DreamGaussian: Fast 3D generation (optimized config)
4. CLIP Validation: Pre-submission quality check

Target: <22 seconds per generation with >0.62 CLIP score
"""

from io import BytesIO
from fastapi import FastAPI, Depends, Form
from fastapi.responses import Response
import uvicorn
import argparse
import time
from loguru import logger
import torch
from PIL import Image
import gc
import numpy as np

from omegaconf import OmegaConf

# SOTA models
from models.flux_generator import FluxImageGenerator
from models.background_remover import SOTABackgroundRemover
from validators.clip_validator import CLIPValidator

# COMMENTED OUT: DreamGaussian (keeping as fallback option)
# from DreamGaussianLib import ModelsPreLoader
# from DreamGaussianLib.GaussianProcessor import GaussianProcessor

# NEW: Microservice-based pipeline for competitive <25s generation
# InstantMesh runs in separate microservice (port 10007) with isolated dependencies
from models.mesh_to_gaussian import MeshToGaussianConverter
import httpx  # For calling InstantMesh microservice
import trimesh  # For loading PLY mesh from InstantMesh
import io  # For BytesIO with PLY data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
    parser.add_argument(
        "--config",
        default="configs/text_mv_fast.yaml",
        help="DreamGaussian config (use fast config for competition)"
    )
    parser.add_argument(
        "--flux-steps",
        type=int,
        default=4,
        help="FLUX.1-schnell inference steps (4 is optimal for schnell variant)"
    )
    parser.add_argument(
        "--validation-threshold",
        type=float,
        default=0.6,
        help="CLIP threshold for validation (0.6 = network minimum)"
    )
    parser.add_argument(
        "--enable-validation",
        action="store_true",
        default=False,
        help="Enable CLIP validation (recommended for competition)"
    )
    return parser.parse_args()


args = get_args()
app = FastAPI(title="404-GEN Competitive Miner")


# Global state
class AppState:
    """Holds all loaded models"""
    flux_generator: FluxImageGenerator = None
    background_remover: SOTABackgroundRemover = None
    # gaussian_models: list = None  # DreamGaussian (commented out)
    instantmesh_service_url: str = "http://localhost:10007"  # Microservice URL
    mesh_to_gaussian: MeshToGaussianConverter = None     # NEW
    clip_validator: CLIPValidator = None


app.state = AppState()


def cleanup_memory():
    """Aggressively clean up GPU memory to prevent OOM errors"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force garbage collection again after clearing cache
        gc.collect()


def precompile_gsplat():
    """
    Pre-compile gsplat CUDA extensions before service starts.

    This ensures the CUDA kernels are compiled at startup rather than
    during the first generation request. If compilation fails, we clear
    the cache and retry once.
    """
    import os
    import shutil
    from pathlib import Path

    logger.info("\n[Pre-check] Compiling gsplat CUDA extensions...")

    try:
        # Import gsplat to trigger JIT compilation
        import gsplat
        from gsplat import rasterization

        logger.info("✅ gsplat CUDA extensions compiled successfully")
        return True

    except Exception as e:
        logger.error(f"❌ gsplat compilation failed: {e}")

        # Try clearing cache and retry once
        cache_dir = Path.home() / ".cache" / "torch_extensions"
        if cache_dir.exists():
            logger.info(f"Clearing torch extensions cache: {cache_dir}")
            shutil.rmtree(cache_dir, ignore_errors=True)

        # Retry import
        try:
            import gsplat
            from gsplat import rasterization
            logger.info("✅ gsplat compiled successfully after cache clear")
            return True
        except Exception as e2:
            logger.error(f"❌ gsplat compilation failed after retry: {e2}")
            raise RuntimeError(
                "Cannot compile gsplat CUDA extensions. "
                "Check CUDA_HOME is set and nvcc is available."
            ) from e2


@app.on_event("startup")
def startup_event():
    """
    Load all models on startup.

    This takes ~30-60 seconds but only happens once.
    """
    logger.info("=" * 60)
    logger.info("404-GEN COMPETITIVE MINER - STARTUP")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Pre-compile gsplat CUDA extensions
    precompile_gsplat()

    # 1. Load Stable Diffusion 1.5 (text-to-image)
    logger.info("\n[1/4] Loading Stable Diffusion 1.5 (text-to-image)...")
    app.state.flux_generator = FluxImageGenerator(device=device)
    logger.info("✅ Stable Diffusion 1.5 ready")

    # Pre-load FLUX to eliminate lazy loading overhead
    logger.info("Pre-loading FLUX to GPU to eliminate lazy loading overhead...")
    try:
        app.state.flux_generator._load_pipeline()
        logger.info("✅ FLUX pre-loaded and ready (eliminates ~2-3s overhead per generation)")
    except Exception as e:
        logger.warning(f"⚠️  FLUX pre-load failed (will lazy load): {e}")

    # 2. Load BRIA RMBG 2.0 (background removal)
    logger.info("\n[2/4] Loading BRIA RMBG 2.0 (background removal)...")
    app.state.background_remover = SOTABackgroundRemover(device=device)
    logger.info("✅ BRIA RMBG 2.0 ready")

    # 3. Check InstantMesh microservice (running separately on port 10007)
    logger.info("\n[3/5] Checking InstantMesh microservice...")
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{app.state.instantmesh_service_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("status") == "healthy":
                    logger.info(f"✅ InstantMesh microservice ready at {app.state.instantmesh_service_url}")
                else:
                    logger.warning(f"⚠️  InstantMesh microservice not ready: {health_data}")
            else:
                logger.error(f"❌ InstantMesh microservice health check failed: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ InstantMesh microservice not available: {e}")
        logger.error("Ensure the service is running: pm2 start instantmesh.config.js")
        raise

    # 4. Initialize Mesh-to-Gaussian converter
    logger.info("\n[4/5] Initializing Mesh-to-Gaussian converter...")
    app.state.mesh_to_gaussian = MeshToGaussianConverter(num_gaussians=8000, base_scale=0.005)
    logger.info("✅ Mesh-to-Gaussian converter ready (3-5s conversion)")

    # FALLBACK: DreamGaussian code (keep commented for easy rollback)
    # logger.info("\n[3/4] Loading DreamGaussian (3D generation)...")
    # config = OmegaConf.load(args.config)
    # app.state.gaussian_models = ModelsPreLoader.preload_model(config, device)
    # logger.info(f"✅ DreamGaussian ready (on {device}, optimized config)")

    # 5. Load CLIP validator (if enabled)
    if args.enable_validation:
        logger.info("\n[5/5] Loading CLIP validator...")
        app.state.clip_validator = CLIPValidator(
            device=device,
            threshold=args.validation_threshold
        )
        logger.info(f"✅ CLIP validator ready (threshold={args.validation_threshold})")
    else:
        logger.warning("\n[5/5] CLIP validation DISABLED (not recommended)")
        app.state.clip_validator = None

    logger.info("\n" + "=" * 60)
    logger.info("🚀 COMPETITIVE MINER READY FOR PRODUCTION")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"SD 1.5 steps: {args.flux_steps}")
    logger.info(f"Validation: {'ON' if args.enable_validation else 'OFF'}")
    logger.info(f"Expected speed: 22-26 seconds per generation")
    logger.info("=" * 60 + "\n")


@app.post("/generate/")
async def generate(prompt: str = Form()) -> Response:
    """
    Competitive generation pipeline.

    Pipeline:
    1. Stable Diffusion 1.5: prompt → image (4s)
    2. rembg: image → RGBA (1s)
    3. DreamGaussian: RGBA → Gaussian Splat (15s)
    4. CLIP Validation: quality check (0.5s)

    Total: ~21 seconds
    """
    t_start = time.time()

    logger.info(f"🎯 Generation request: '{prompt}'")

    try:
        # Clean memory before starting generation
        cleanup_memory()

        # Step 1: Text-to-image with Stable Diffusion 1.5
        t1 = time.time()
        logger.info("  [1/4] Generating image with Stable Diffusion 1.5...")

        # SKIP: Keep DreamGaussian on GPU (RTX 4090 has 24GB VRAM)
        # Note: Moving models between CPU/GPU causes tensor device mismatches
        # logger.debug("  Moving DreamGaussian models to CPU to free GPU for SD...")
        # for model in app.state.gaussian_models:
        #     if hasattr(model, 'to'):
        #         model.to('cpu')

        # Cleanup memory without moving models
        torch.cuda.synchronize()
        cleanup_memory()

        logger.debug("  Memory cleaned, loading SD to GPU...")

        # Now move SD to GPU
        app.state.flux_generator.ensure_on_gpu()

        # CUDA sync before generation
        torch.cuda.synchronize()

        # Enhance prompt for better 2D image generation (not 3D terms!)
        enhanced_prompt = f"{prompt}, highly detailed, professional photo, centered object, white background, studio lighting"

        # Use 512x512 for better CLIP scores (CLIP prefers higher resolution)
        image = app.state.flux_generator.generate(
            prompt=enhanced_prompt,
            num_inference_steps=args.flux_steps,
            height=512,
            width=512
        )

        t2 = time.time()
        logger.info(f"  ✅ SD 1.5 done ({t2-t1:.2f}s)")

        # Aggressively offload SD to free GPU memory for next stage
        app.state.flux_generator.offload_to_cpu()
        cleanup_memory()
        logger.debug(f"  GPU memory freed after SD 1.5")

        # Step 2: Background removal with rembg
        logger.info("  [2/4] Removing background with rembg...")

        # Note: Background remover automatically moves to GPU and back to CPU
        rgba_image = app.state.background_remover.remove_background(
            image,
            threshold=0.5
        )

        t3 = time.time()
        logger.info(f"  ✅ Background removal done ({t3-t2:.2f}s)")

        # Free the input image from memory
        del image
        cleanup_memory()
        logger.debug(f"  GPU memory freed after background removal")

        # Step 3: 3D mesh generation with InstantMesh microservice (1-2s)
        t3_start = time.time()
        logger.info("  [3/5] Generating 3D mesh with InstantMesh microservice...")

        try:
            # Convert RGBA image to base64 for HTTP transfer
            import io
            import base64
            buffer = io.BytesIO()
            rgba_image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Call InstantMesh microservice
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{app.state.instantmesh_service_url}/generate_mesh",
                    json={
                        "image_base64": image_base64,
                        "scale": 1.0,
                        "input_size": 320
                    }
                )

                if response.status_code != 200:
                    raise RuntimeError(f"InstantMesh service returned status {response.status_code}: {response.text}")

                result = response.json()

            # Decode mesh from base64
            mesh_bytes = base64.b64decode(result["mesh_base64"])
            mesh = trimesh.load(io.BytesIO(mesh_bytes), file_type='ply')

            t4 = time.time()
            logger.info(f"  ✅ Mesh generation done ({t4-t3_start:.2f}s, service: {result['generation_time']:.2f}s)")
            logger.info(f"     Mesh quality: {result['num_vertices']} vertices, {result['num_faces']} faces")

        except Exception as e:
            logger.error(f"InstantMesh microservice call failed: {e}", exc_info=True)
            raise

        # Step 4: Convert mesh to Gaussian Splat PLY (3-5s)
        t4_start = time.time()
        logger.info("  [4/5] Converting mesh to Gaussian Splat...")

        try:
            # Convert mesh to Gaussian Splat PLY (subnet-compatible format)
            ply_buffer = app.state.mesh_to_gaussian.convert(
                mesh,
                num_gaussians=8000  # Target number of Gaussian splats
            )
            ply_bytes = ply_buffer.getvalue()

            # Clean up mesh from memory
            del mesh
            cleanup_memory()

            t5 = time.time()
            logger.info(f"  ✅ Mesh-to-Gaussian conversion done ({t5-t4_start:.2f}s)")
            logger.info(f"     Generated {len(ply_bytes)/1024:.1f} KB Gaussian Splat PLY")

        except Exception as e:
            logger.error(f"Mesh-to-Gaussian conversion failed: {e}", exc_info=True)
            cleanup_memory()
            raise

        # FALLBACK: DreamGaussian code (keep commented for easy rollback)
        # import tempfile, os
        # with tempfile.NamedTemporaryFile(suffix='_rgba.png', delete=False) as tmp:
        #     rgba_image.save(tmp.name)
        #     tmp_path = tmp.name
        # caption_path = tmp_path.replace('_rgba.png', '_caption.txt')
        # with open(caption_path, 'w') as f:
        #     f.write(prompt)
        # try:
        #     config = OmegaConf.load(args.config)
        #     config.input = tmp_path
        #     config.prompt = prompt
        #     gaussian_processor = GaussianProcessor(config, prompt)
        #     gaussian_processor.train(app.state.gaussian_models, config.iters)
        #     buffer = BytesIO()
        #     gaussian_processor.get_gs_model().save_ply(buffer)
        #     buffer.seek(0)
        #     ply_bytes = buffer.getvalue()
        # finally:
        #     if os.path.exists(tmp_path):
        #         os.remove(tmp_path)
        #     if os.path.exists(caption_path):
        #         os.remove(caption_path)
        # t4 = time.time()
        # logger.info(f"  ✅ 3D generation done ({t4-t3:.2f}s)")

        # Step 5: CLIP validation
        if app.state.clip_validator:
            logger.info("  [5/5] Validating with CLIP...")

            # TEMPORARY: Validate the 2D RGBA image instead of rendering 3D
            # (3D rendering has persistent OOM issues that need deeper investigation)
            logger.warning("  Using 2D image validation (temporary workaround)")

            # Convert RGBA to RGB for CLIP
            validation_image = rgba_image.convert("RGB")

            # Validate with CLIP
            app.state.clip_validator.to_gpu()
            passes, score = app.state.clip_validator.validate_image(
                validation_image,
                prompt
            )
            app.state.clip_validator.to_cpu()

            t6 = time.time()
            logger.info(f"  ✅ Validation done ({t6-t5:.2f}s)")
            logger.info(f"  2D Image CLIP score: {score:.3f}")

            # Clean up
            del validation_image
            cleanup_memory()

        # GPU memory cleaned up after validation
        cleanup_memory()
        logger.debug(f"  GPU memory freed after 3D generation and validation")

        # Check if validation passed
        if app.state.clip_validator and not passes:
            logger.warning(
                f"  ⚠️  VALIDATION FAILED: CLIP={score:.3f} < {args.validation_threshold}"
            )
            logger.warning("  Returning empty result to avoid cooldown penalty")

            # Return empty PLY instead of bad result
            empty_buffer = BytesIO()
            return Response(
                empty_buffer.getvalue(),
                media_type="application/octet-stream",
                headers={"X-Validation-Failed": "true", "X-CLIP-Score": str(score)}
            )

        if app.state.clip_validator:
            logger.info(f"  ✅ VALIDATION PASSED: CLIP={score:.3f}")

        # Success!
        t_total = time.time() - t_start

        logger.info("=" * 60)
        logger.info(f"✅ GENERATION COMPLETE")
        logger.info(f"   Total time: {t_total:.2f}s")
        logger.info(f"   FLUX (2-step): {t2-t1:.2f}s")
        logger.info(f"   Background: {t3-t2:.2f}s")
        logger.info(f"   InstantMesh: {t4-t3:.2f}s")
        logger.info(f"   Mesh→Gaussian: {t5-t4:.2f}s")
        if app.state.clip_validator:
            logger.info(f"   Validation: {t6-t5:.2f}s")
            logger.info(f"   CLIP Score: {score:.3f}")
        logger.info(f"   File size: {len(ply_bytes)/1024:.1f} KB")
        logger.info("=" * 60)

        return Response(
            ply_bytes,
            media_type="application/octet-stream",
            headers={
                "X-Generation-Time": str(t_total),
                "X-CLIP-Score": str(score) if app.state.clip_validator else "N/A"
            }
        )

    except Exception as e:
        import traceback
        logger.error(f"❌ Generation failed: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")

        # Clean up memory on error
        cleanup_memory()

        # Return empty result on error (better than crash)
        empty_buffer = BytesIO()
        return Response(
            empty_buffer.getvalue(),
            media_type="application/octet-stream",
            headers={"X-Generation-Error": str(e)},
            status_code=500
        )
    finally:
        # Always clean up memory after request completes
        cleanup_memory()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "flux": app.state.flux_generator is not None,
            "background_remover": app.state.background_remover is not None,
            "instantmesh_service": app.state.instantmesh_service_url,  # Microservice URL
            "mesh_to_gaussian": app.state.mesh_to_gaussian is not None,
            # "dreamgaussian": app.state.gaussian_models is not None,  # Commented out
            "clip_validator": app.state.clip_validator is not None
        },
        "config": {
            "flux_steps": args.flux_steps,
            "validation_enabled": args.enable_validation,
            "validation_threshold": args.validation_threshold if args.enable_validation else None
        }
    }


@app.get("/stats")
async def get_stats():
    """Get generation statistics"""
    # Could add metrics tracking here
    return {
        "service": "404-gen-competitive-miner",
        "version": "1.0.0",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


if __name__ == "__main__":
    logger.info("Starting 404-GEN Competitive Generation Service...")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="info"
    )
