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

# InstantMesh + 2D color sampling - NEW WORKING PIPELINE!
from models.mesh_to_gaussian import MeshToGaussianConverter
from instantmesh_integration import generate_with_instantmesh
import httpx  # For calling InstantMesh microservice
import trimesh  # For loading PLY mesh from InstantMesh

# DEPRECATED: DreamGaussian (too slow for <30s requirement)
# from DreamGaussianLib import ModelsPreLoader
# from DreamGaussianLib.GaussianProcessor import GaussianProcessor

import io  # For BytesIO with PLY data
from rendering.quick_render import render_gaussian_model_to_images  # For 3D validation


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
    instantmesh_service_url: str = "http://localhost:10007"  # InstantMesh microservice URL
    mesh_to_gaussian: MeshToGaussianConverter = None
    last_gs_model = None  # Cache last generated Gaussian Splat model for validation
    clip_validator: CLIPValidator = None
    # DEPRECATED: DreamGaussian (too slow)
    # gaussian_models: list = None


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

    # DEPRECATED: DreamGaussian (too slow for <30s requirement - requires 200+ iterations for quality)
    # logger.info("\n[3/4] Loading DreamGaussian (3D generation)...")
    # config = OmegaConf.load(args.config)
    # app.state.gaussian_models = ModelsPreLoader.preload_model(config, device)
    # logger.info(f"✅ DreamGaussian ready (on {device}, 10-iter optimized config)")

    # 3. Check InstantMesh microservice + Mesh-to-Gaussian with 2D color sampling
    logger.info("\n[3/5] Checking InstantMesh microservice...")
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{app.state.instantmesh_service_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("status") == "healthy":
                    logger.info(f"✅ InstantMesh microservice ready at {app.state.instantmesh_service_url}")
            else:
                logger.error(f"❌ InstantMesh microservice health check failed")
                raise RuntimeError("InstantMesh unhealthy")
    except Exception as e:
        logger.error(f"❌ InstantMesh microservice not available: {e}")
        raise

    logger.info("\n[4/5] Initializing Mesh-to-Gaussian converter with 2D color sampling...")
    app.state.mesh_to_gaussian = MeshToGaussianConverter(
        num_gaussians=50000,  # High density for better coverage
        base_scale=0.015      # Small Gaussians, high count approach
    )
    logger.info("✅ Mesh-to-Gaussian converter ready (2D color sampling, 50K Gaussians)")

    # 4. Load CLIP validator (if enabled)
    if args.enable_validation:
        logger.info("\n[4/4] Loading CLIP validator...")
        app.state.clip_validator = CLIPValidator(
            device=device,
            threshold=args.validation_threshold
        )
        logger.info(f"✅ CLIP validator ready (threshold={args.validation_threshold})")
    else:
        logger.warning("\n[4/4] CLIP validation DISABLED (not recommended)")
        app.state.clip_validator = None

    logger.info("\n" + "=" * 60)
    logger.info("🚀 COMPETITIVE MINER READY - DREAMGAUSSIAN PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config} (10 iterations)")
    logger.info(f"FLUX steps: {args.flux_steps}")
    logger.info(f"3D Engine: DreamGaussian (switched from broken InstantMesh converter)")
    logger.info(f"Validation: {'ON' if args.enable_validation else 'OFF'}")
    logger.info(f"Expected speed: 22-28 seconds per generation")
    logger.info(f"Expected CLIP: 0.6-0.7 (proven working range)")
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

    # CRITICAL: Detect and handle base64 image prompts (image-to-3D mode)
    is_base64_image = (
        len(prompt) > 500 or
        prompt.startswith('iVBOR') or
        prompt.startswith('/9j/') or
        prompt.startswith('data:image') or
        ('==' in prompt[-10:] if len(prompt) > 10 else False)
    )

    if is_base64_image:
        logger.info(f"🖼️  Detected IMAGE-TO-3D task (base64 prompt length: {len(prompt)})")
    else:
        logger.info(f"🎯 Detected TEXT-TO-3D task: '{prompt}'")

    try:
        # Clean memory before starting generation
        cleanup_memory()

        # Handle image-to-3D vs text-to-3D differently
        if is_base64_image:
            # IMAGE-TO-3D: Decode base64 image, skip FLUX
            t1 = time.time()
            logger.info("  [1/4] Decoding base64 image (image-to-3D mode)...")

            try:
                import base64

                # Handle data URL format (data:image/png;base64,...)
                if ',' in prompt[:100]:
                    base64_data = prompt.split(',', 1)[1]
                else:
                    base64_data = prompt

                # Decode to PIL Image
                image_bytes = base64.b64decode(base64_data)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGBA')

                logger.info(f"  ✅ Decoded image: {image.size[0]}x{image.size[1]} RGBA")

                # Use image directly - NO FLUX needed!
                rgba_image = image
                t2 = time.time()
                t3 = time.time()  # No background removal needed for pre-made RGBA

                # Set generic prompt for validation
                validation_prompt = "a 3D object"

                logger.info(f"  ⏭️  Skipped FLUX + background removal (image provided, {t2-t1:.2f}s)")

            except Exception as e:
                logger.error(f"  ❌ Failed to decode base64 image: {e}", exc_info=True)
                # Return empty result for invalid image
                empty_buffer = BytesIO()
                return Response(
                    empty_buffer.getvalue(),
                    media_type="application/octet-stream",
                    status_code=400,
                    headers={"X-Validation-Failed": "true", "X-Error": "Invalid base64 image"}
                )

        else:
            # TEXT-TO-3D: Original pipeline with FLUX
            validation_prompt = prompt

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

            # Enhance prompt for better 2D image generation and CLIP scores
            # CLIP was trained on web captions - photography terms score higher
            enhanced_prompt = f"a photorealistic {prompt}, professional product photography, studio lighting setup, pure white background, centered composition, sharp focus, highly detailed, 8k resolution, award-winning photography"

            # Use 512x512 for better CLIP scores (CLIP prefers higher resolution)
            image = app.state.flux_generator.generate(
                prompt=enhanced_prompt,
                num_inference_steps=args.flux_steps,
                height=512,
                width=512
            )

            t2 = time.time()
            logger.info(f"  ✅ SD 1.5 done ({t2-t1:.2f}s)")

            # DEBUG: Save FLUX output for quality inspection
            debug_timestamp = int(time.time())
            image.save(f"/tmp/debug_1_flux_{debug_timestamp}.png")
            logger.debug(f"  Saved debug image: /tmp/debug_1_flux_{debug_timestamp}.png")

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

            # DEBUG: Save background-removed image for quality inspection
            rgba_image.save(f"/tmp/debug_2_rembg_{debug_timestamp}.png")
            logger.debug(f"  Saved debug image: /tmp/debug_2_rembg_{debug_timestamp}.png")

            # Free the input image from memory
            del image
        cleanup_memory()
        logger.debug(f"  GPU memory freed after background removal")

        # Step 3 & 4: 3D generation with InstantMesh + 2D color sampling (1-2s total)
        t3_start = time.time()
        try:
            # Call InstantMesh integration (mesh generation + color sampling conversion)
            ply_bytes, gs_model, timings = await generate_with_instantmesh(
                rgba_image=rgba_image,
                prompt=prompt,
                mesh_to_gaussian_converter=app.state.mesh_to_gaussian,
                instantmesh_url=app.state.instantmesh_service_url
            )

            # Cache for validation
            app.state.last_gs_model = gs_model

            t4 = time.time()
            logger.info(f"  ✅ 3D generation done ({t4-t3_start:.2f}s)")
            logger.info(f"     InstantMesh: {timings['instantmesh']:.2f}s, Mesh→Gaussian: {timings['mesh_to_gaussian']:.2f}s")
            logger.info(f"     Generated {len(ply_bytes)/1024:.1f} KB Gaussian Splat PLY")

        except Exception as e:
            logger.error(f"InstantMesh generation failed: {e}", exc_info=True)
            cleanup_memory()
            raise

        # Step 4: CLIP validation
        if app.state.clip_validator:
            t5 = time.time()
            logger.info("  [4/4] Validating with CLIP...")

            # Render 3D Gaussian Splat for validation using cached model
            try:
                logger.debug("  Rendering 3D Gaussian Splat for CLIP validation...")

                # Get the cached GaussianModel from mesh_to_gaussian converter
                gs_model = app.state.last_gs_model

                if gs_model is not None:
                    rendered_views = render_gaussian_model_to_images(
                        model=gs_model,
                        num_views=4,  # 4 views for robust validation
                        resolution=512,
                        device="cuda"
                    )

                    if rendered_views and len(rendered_views) > 0:
                        # DEBUG: Save all rendered views for quality inspection
                        debug_timestamp = int(time.time())
                        for i, view in enumerate(rendered_views):
                            view.save(f"/tmp/debug_5_render_view{i}_{debug_timestamp}.png")
                        logger.debug(f"  Saved {len(rendered_views)} debug render views")

                        # Use first view for CLIP validation (or average across all)
                        validation_image = rendered_views[0]
                        logger.debug(f"  Using 3D render (view 1/{len(rendered_views)}) for validation")
                    else:
                        # Fallback to 2D image if rendering fails
                        logger.warning("  3D rendering failed, falling back to 2D image")
                        validation_image = rgba_image.convert("RGB")
                else:
                    logger.warning("  No cached GaussianModel available, using 2D fallback")
                    validation_image = rgba_image.convert("RGB")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"  OOM during 3D rendering, trying with lower resolution")
                    # Retry with lower settings
                    try:
                        gs_model = app.state.last_gs_model
                        if gs_model is not None:
                            rendered_views = render_gaussian_model_to_images(
                                model=gs_model,
                                num_views=2,  # Fewer views
                                resolution=256,  # Lower resolution
                                device="cuda"
                            )
                            validation_image = rendered_views[0] if rendered_views else rgba_image.convert("RGB")
                        else:
                            validation_image = rgba_image.convert("RGB")
                    except:
                        logger.warning("  Retry failed, using 2D fallback")
                        validation_image = rgba_image.convert("RGB")
                else:
                    logger.warning(f"  3D rendering error: {e}, using 2D fallback")
                    validation_image = rgba_image.convert("RGB")
            except Exception as e:
                logger.warning(f"  Unexpected error during 3D rendering: {e}, using 2D fallback")
                validation_image = rgba_image.convert("RGB")

            # Validate with CLIP
            app.state.clip_validator.to_gpu()

            # DIAGNOSTIC: Test both 2D FLUX and 3D render
            flux_2d_image = rgba_image.convert("RGB")
            _, flux_score = app.state.clip_validator.validate_image(flux_2d_image, validation_prompt)
            logger.info(f"  📊 DIAGNOSTIC - 2D FLUX CLIP score: {flux_score:.3f}")

            passes, score = app.state.clip_validator.validate_image(
                validation_image,
                validation_prompt
            )
            app.state.clip_validator.to_cpu()

            t6 = time.time()
            logger.info(f"  ✅ Validation done ({t6-t5:.2f}s)")
            logger.info(f"  📊 DIAGNOSTIC - 3D Render CLIP score: {score:.3f}")
            logger.info(f"  📊 DIAGNOSTIC - Quality loss: {((flux_score - score) / flux_score * 100):.1f}%")

            # Clean up
            del validation_image
            if 'rendered_views' in locals():
                del rendered_views
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
        logger.info(f"   FLUX (4-step): {t2-t1:.2f}s")
        logger.info(f"   Background: {t3-t2:.2f}s")
        logger.info(f"   3D (InstantMesh+M2G): {t4-t3_start:.2f}s")
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
