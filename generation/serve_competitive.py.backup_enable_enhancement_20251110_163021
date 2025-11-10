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
import asyncio

from omegaconf import OmegaConf

# SOTA models
from models.sdxl_turbo_generator import SDXLTurboGenerator
from models.background_remover import SOTABackgroundRemover
from validators.clip_validator import CLIPValidator

# TRELLIS Native Gaussian Generation - PRODUCTION PIPELINE!
# 256K gaussians, 5s generation, 16.6 MB files
from trellis_integration import generate_with_trellis
import httpx  # For calling TRELLIS microservice

# DEPRECATED: InstantMesh + 2D color sampling (50K gaussians, insufficient for mainnet)
# from models.mesh_to_gaussian import MeshToGaussianConverter
# from instantmesh_integration import generate_with_instantmesh
# import trimesh  # For loading PLY mesh from InstantMesh

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
        help="SD3.5 Large Turbo inference steps (4 is optimal for turbo variant)"
    )
    parser.add_argument(
        "--validation-threshold",
        type=float,
        default=0.20,
        help="CLIP threshold for validation (0.20 = safe cushion, raw CLIP scores range 0.15-0.35)"
    )
    parser.add_argument(
        "--enable-validation",
        action="store_true",
        default=False,
        help="Enable CLIP validation (recommended for competition)"
    )
    parser.add_argument(
        "--enable-scale-normalization",
        action="store_true",
        default=False,
        help="Enable scale normalization correction (diagnostic mode: default OFF)"
    )
    parser.add_argument(
        "--enable-prompt-enhancement",
        action="store_true",
        default=False,
        help="Enable prompt enhancement with quality hints (diagnostic mode: default OFF)"
    )
    parser.add_argument(
        "--enable-image-enhancement",
        action="store_true",
        default=False,
        help="Enable image enhancement before TRELLIS (diagnostic mode: default OFF)"
    )
    parser.add_argument(
        "--min-gaussian-count",
        type=int,
        default=0,
        help="Minimum gaussian count threshold (0 = disabled, 150000 = strict quality gate)"
    )
    parser.add_argument(
        "--background-threshold",
        type=float,
        default=0.5,
        help="Background removal threshold (0.5 = rembg default, 0.3 = softer edges)"
    )
    return parser.parse_args()


# Fix for uvicorn import: conditional args parsing
# When uvicorn imports this module, it passes its own args, causing argparse to fail.
# Solution: Only parse args when running directly, use defaults when imported.
if __name__ == "__main__":
    args = get_args()
else:
    # Uvicorn import path - use default config
    class Args:
        port = 10010
        config = "configs/text_mv_fast.yaml"
        flux_steps = 4
        validation_threshold = 0.20
        enable_validation = False
        enable_scale_normalization = False
        enable_prompt_enhancement = False  # DISABLED: Phase 1A - prompt enhancement reduces density by 5.7%
        enable_image_enhancement = False
        min_gaussian_count = 150000  # ENABLED: Phase 2A - filter LOW density models (<150K = 50% acceptance)
        background_threshold = 0.5  # Standard rembg threshold, preserves more object detail
    args = Args()

app = FastAPI(title="404-GEN Competitive Miner")


# Global state
class AppState:
    """Holds all loaded models"""
    flux_generator: SDXLTurboGenerator = None  # SDXL-Turbo on GPU 1 (RTX 5070 Ti)  # Stable Cascade on GPU 1 (RTX 5070 Ti)
    background_remover: SOTABackgroundRemover = None
    generation_semaphore: asyncio.Semaphore = None  # Limit to 1 concurrent generation
    trellis_service_url: str = "http://localhost:10008"  # TRELLIS microservice URL
    # DEPRECATED: InstantMesh + Mesh-to-Gaussian (50K gaussians insufficient for mainnet)
    # instantmesh_service_url: str = "http://localhost:10007"
    # mesh_to_gaussian: MeshToGaussianConverter = None
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


def enhance_prompt_for_detail(prompt):
    """
    Enhance prompts to increase SDXL-Turbo image complexity and TRELLIS gaussian density.

    Strategy:
    1. Detect sparse prompts (short, simple)
    2. Add volumetric/detail hints to bias SDXL toward complex rendering
    3. Add material/texture hints to increase surface complexity
    4. Add lighting hints to increase depth cues

    Theory: More detailed prompts → More complex images → More gaussians → Higher success rate

    Target: Push sparse prompts from 150-250K gaussians → 400K+ gaussians (86% success tier)
    Expected impact: +20-25% overall success rate (60% → 80-85%)
    """
    import random

    # Detect sparse prompts (word count < 8 = likely to generate <250K gaussians)
    word_count = len(prompt.split())

    # Enhancement hint libraries
    detail_hints = [
        "intricate details, complex geometry, volumetric forms",
        "elaborate design, rich surface detail, dimensional depth",
        "fine craftsmanship, textured surfaces, layered complexity",
        "detailed construction, ornate features, multi-part assembly"
    ]

    material_hints = [
        "high-quality materials, varied textures, surface variation",
        "premium finish, tactile details, material complexity",
        "refined surfaces, texture contrast, material depth",
        "quality craftsmanship, surface richness, textural detail"
    ]

    lighting_hints = [
        "professional studio lighting, volumetric shadows, depth definition",
        "dramatic lighting, dimensional shadows, form-revealing illumination",
        "sculptural lighting, shadow detail, 3D depth emphasis",
        "gallery lighting, form-defining shadows, volumetric depth"
    ]

    background_hints = [
        "solid dark background, clean backdrop, isolated composition",
        "black studio background, centered focus, negative space",
        "neutral gray backdrop, professional product shot, clean separation",
        "dark studio setting, focused composition, clear silhouette"
    ]

    # Apply enhancement based on sparsity
    if word_count < 8:
        # AGGRESSIVE ENHANCEMENT for sparse prompts (high rejection risk)
        enhancement = f"{random.choice(detail_hints)}, {random.choice(material_hints)}, {random.choice(lighting_hints)}, {random.choice(background_hints)}"
        logger.debug(f"  🎯 SPARSE PROMPT DETECTED ({word_count} words) - applying aggressive enhancement")
    else:
        # LIGHT ENHANCEMENT for complex prompts (already likely to generate high density)
        enhancement = f"{random.choice(detail_hints)}, {random.choice(background_hints)}"
        logger.debug(f"  ✓ Complex prompt ({word_count} words) - applying light enhancement")

    return f"{prompt}, {enhancement}"


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

    # Initialize concurrency limiter (prevents TRELLIS queueing)
    app.state.generation_semaphore = asyncio.Semaphore(1)
    logger.info("🔒 Generation concurrency limit: 1 (prevents TRELLIS queueing)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Pre-compile gsplat CUDA extensions
    precompile_gsplat()

    # 1. Initialize Stable Cascade generator (LAZY LOADING, GPU 1)
    logger.info("\n[1/4] Initializing SDXL-Turbo (lazy loading)...")
    app.state.flux_generator = SDXLTurboGenerator(device="cuda:1")  # GPU 1: Stable Cascade
    logger.info("✅ SDXL-Turbo initialized (will load on first request)")
    logger.info("   Multi-GPU setup:")
    logger.info("     - GPU 0 (RTX 4090, 24GB): TRELLIS + Background removal (~6GB)")
    logger.info("     - GPU 1 (RTX 5070 Ti, 15.47GB): SDXL-Turbo (~4-5GB, 11GB free!)")
    logger.info("   Speed: ~1-2s image generation (13x faster than FLUX!)")
    logger.info("   Architecture: Prior (5.1GB) + Decoder (1.5GB)")

    # 2. Load BRIA RMBG 2.0 (background removal) - FORCE TO GPU 0 TO KEEP GPU 1 FREE FOR SD3.5
    logger.info("\n[2/4] Loading BRIA RMBG 2.0 (background removal)...")
    app.state.background_remover = SOTABackgroundRemover(device="cuda:0")  # GPU 0: 0.15s, ~1-2GB (shares with TRELLIS)
    logger.info("✅ BRIA RMBG 2.0 ready (GPU 0 - keeps GPU 1 free for SD3.5)")

    # DEPRECATED: DreamGaussian (too slow for <30s requirement - requires 200+ iterations for quality)
    # logger.info("\n[3/4] Loading DreamGaussian (3D generation)...")
    # config = OmegaConf.load(args.config)
    # app.state.gaussian_models = ModelsPreLoader.preload_model(config, device)
    # logger.info(f"✅ DreamGaussian ready (on {device}, 10-iter optimized config)")

    # 3. Check TRELLIS microservice for native Gaussian generation
    logger.info("\n[3/5] Checking TRELLIS microservice...")
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{app.state.trellis_service_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("status") == "healthy":
                    logger.info(f"✅ TRELLIS microservice ready at {app.state.trellis_service_url}")
                    logger.info(f"   Pipeline: {health_data.get('pipeline_loaded', 'Unknown')}")
            else:
                logger.warning(f"⚠️ TRELLIS microservice returned non-200 status")
                logger.warning("  Worker will start anyway, but generations will fail until TRELLIS is up")
    except Exception as e:
        logger.warning(f"⚠️ TRELLIS microservice not available: {e}")
        logger.warning("  Worker will start anyway, but generations will fail until TRELLIS is up")
        logger.warning("  Requests will return 503 Service Unavailable until TRELLIS recovers")

    # DEPRECATED: Mesh-to-Gaussian converter (InstantMesh approach with 50K gaussians)
    # TRELLIS generates native gaussians directly - no mesh conversion needed!
    # logger.info("\n[4/5] Initializing Mesh-to-Gaussian converter with 2D color sampling...")
    # app.state.mesh_to_gaussian = MeshToGaussianConverter(
    #     num_gaussians=50000,  # High density for better coverage
    #     base_scale=0.015      # Small Gaussians, high count approach
    # )
    # logger.info("✅ Mesh-to-Gaussian converter ready (2D color sampling, 50K Gaussians)")

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
    logger.info("🚀 COMPETITIVE MINER READY - SDXL-TURBO + TRELLIS")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Image Generator: SDXL-Turbo (1-4 steps)")
    logger.info(f"Image Generator Quality: 80-85% of FLUX, proven and stable")
    logger.info(f"3D Engine: TRELLIS (native gaussian splat generation)")
    logger.info(f"Validation: {'ON' if args.enable_validation else 'OFF'}")
    logger.info(f"Expected speed: 13-17 seconds per generation")
    logger.info(f"Expected CLIP: 0.55-0.70 (SDXL-Turbo for mining speed)")
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

    # Acquire semaphore to limit concurrent generations (prevents TRELLIS queueing)
    async with app.state.generation_semaphore:
        logger.debug("🔒 Acquired generation lock")

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

                    # Set prompt stats for image-to-3D mode (no text prompt to enhance)
                    original_words = 0
                    enhanced_words = 0

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
                logger.info("  [1/4] Generating image with SDXL-Turbo...")
    
                # SKIP: Keep DreamGaussian on GPU (RTX 4090 has 24GB VRAM)
                # Note: Moving models between CPU/GPU causes tensor device mismatches
                # logger.debug("  Moving DreamGaussian models to CPU to free GPU for SD...")
                # for model in app.state.gaussian_models:
                #     if hasattr(model, 'to'):
                #         model.to('cpu')
    
                # Cleanup memory without moving models
                torch.cuda.synchronize()
                cleanup_memory()
    
                # SD3.5 stays on GPU permanently - no need to reload!
                # (TRELLIS runs in separate process, no GPU conflict)
    
                # SD3.5 Large Turbo: Better for 3D than FLUX
                # - 8B params with 3 text encoders (T5, CLIP-L, CLIP-G)
                # - Better prompt adherence and depth perception
                # - Expected CLIP scores: 0.60-0.75 (vs FLUX 0.24-0.27)

                # DIAGNOSTIC MODE: Prompt enhancement is OPTIONAL
                # By default (enable_prompt_enhancement=False), we use raw prompts (like official template)
                if args.enable_prompt_enhancement:
                    enhanced_prompt = enhance_prompt_for_detail(prompt)
                    logger.debug(f"  📝 Prompt enhanced: '{prompt}' → '{enhanced_prompt}'")
                else:
                    enhanced_prompt = prompt
                    logger.debug(f"  📝 Using raw prompt (no enhancement): '{prompt}'")

                # MEASUREMENT: Track prompt stats for density correlation analysis
                original_words = len(prompt.split())
                enhanced_words = len(enhanced_prompt.split())
                was_enhanced = args.enable_prompt_enhancement
                logger.info(f"  📏 PROMPT STATS: {original_words}w → {enhanced_words}w (enhanced={was_enhanced})")

                # Use 512x512 for better CLIP scores (CLIP prefers higher resolution)
                image = app.state.flux_generator.generate(
                    prompt=enhanced_prompt,
                    num_inference_steps=args.flux_steps,
                    height=512,
                    width=512
                )
    
                t2 = time.time()
                logger.info(f"  ✅ SDXL-Turbo done ({t2-t1:.2f}s)")
    
                # DEBUG: Save SD3.5 output for quality inspection
                debug_timestamp = int(time.time())
                image.save(f"/tmp/debug_1_sd35_{debug_timestamp}.png")
                logger.debug(f"  Saved debug image: /tmp/debug_1_sd35_{debug_timestamp}.png")

                # FLUX stays loaded on GPU 1 permanently with NF4 quantization (~6-7GB)
                # No unloading needed - TRELLIS runs on separate GPU 0 via microservice
                logger.info(f"  ✅ SDXL-Turbo generation complete (stays loaded on GPU 1)")

                # Step 2: Background removal with rembg
                logger.info("  [2/4] Removing background with rembg...")

                # DIAGNOSTIC MODE: Threshold is configurable
                # default=0.5 (rembg standard), can test 0.3 (softer edges) if needed
                rgba_image = app.state.background_remover.remove_background(
                    image,
                    threshold=args.background_threshold
                )
                logger.debug(f"  Background removal threshold: {args.background_threshold}")
    
                t3 = time.time()
                logger.info(f"  ✅ Background removal done ({t3-t2:.2f}s)")
    
                # DEBUG: Save background-removed image for quality inspection
                rgba_image.save(f"/tmp/debug_2_rembg_{debug_timestamp}.png")
                logger.debug(f"  Saved debug image: /tmp/debug_2_rembg_{debug_timestamp}.png")
    
                # Free the input image from memory
                del image
            cleanup_memory()
            logger.debug(f"  GPU memory freed after background removal")
    
            # Step 3: Native Gaussian generation with TRELLIS (5s, 256K gaussians)
            t3_start = time.time()
            try:
                # Call TRELLIS microservice for direct gaussian splat generation
                ply_bytes, gs_model, timings = await generate_with_trellis(
                    rgba_image=rgba_image,
                    prompt=prompt,
                    trellis_url="http://localhost:10008",
                    enable_scale_normalization=args.enable_scale_normalization,
                    enable_image_enhancement=args.enable_image_enhancement,
                    min_gaussians=args.min_gaussian_count
                )
    
                # Cache for validation
                app.state.last_gs_model = gs_model
    
                t4 = time.time()
                logger.info(f"  ✅ 3D generation done ({t4-t3_start:.2f}s)")
                logger.info(f"     TRELLIS: {timings['trellis']:.2f}s, Model Load: {timings['model_load']:.2f}s")
                logger.info(f"     Generated {len(ply_bytes)/1024:.1f} KB Gaussian Splat PLY")
                logger.info(f"     📊 Generation stats: {timings['num_gaussians']:,} gaussians, {timings['file_size_mb']:.1f}MB")

                # MEASUREMENT: Track prompt-to-density correlation
                density_tier = "HIGH" if timings['num_gaussians'] >= 400000 else ("MED" if timings['num_gaussians'] >= 150000 else "LOW")
                logger.info(f"     📈 DENSITY CORRELATION: {original_words}w → {enhanced_words}w → {timings['num_gaussians']:,}g [{density_tier}]")

                # Phase 1C: Timeout risk filter (35s threshold)
                # Tasks >30s have 9.3% lower acceptance rate (44.9% vs 54.2%)
                # Use 35s as conservative cutoff to avoid validator timeouts
                total_time_so_far = time.time() - t_start
                if total_time_so_far > 35.0:
                    logger.warning(f"  ⚠️ TIMEOUT RISK: Generation took {total_time_so_far:.1f}s > 35s threshold")
                    logger.warning(f"     Tasks >30s have 9.3% lower acceptance (44.9% vs 54.2%)")
                    logger.warning(f"     Filtering to avoid validator timeout penalty")
                    cleanup_memory()
                    # Return empty response to skip this task
                    empty_buffer = BytesIO()
                    return Response(
                        empty_buffer.getvalue(),
                        media_type="application/octet-stream",
                        status_code=200,  # 200 but empty = skip this task
                        headers={"X-Skip-Reason": "Timeout risk - generation >35s"}
                    )

            except ValueError as e:
                # Quality gate: Sparse generation detected
                if "Insufficient gaussian density" in str(e):
                    logger.warning(f"  ⚠️ Generation quality too low, skipping submission to avoid Score=0.0")
                    logger.warning(f"  Details: {e}")
                    cleanup_memory()
                    # Return empty response to skip this task
                    empty_buffer = BytesIO()
                    return Response(
                        empty_buffer.getvalue(),
                        media_type="application/octet-stream",
                        status_code=200,  # 200 but empty = skip this task
                        headers={"X-Skip-Reason": "Quality gate failed - sparse generation"}
                    )
                else:
                    # Other ValueError - re-raise
                    logger.error(f"TRELLIS generation failed: {e}", exc_info=True)
                    cleanup_memory()
                    raise
            except Exception as e:
                import httpx
                if isinstance(e, (httpx.ConnectError, httpx.TimeoutException)):
                    logger.error(f"⚠️ TRELLIS microservice unavailable: {e}")
                    logger.error("  Returning service unavailable instead of submitting garbage")
                    cleanup_memory()
                    # Return 503 Service Unavailable - miner should NOT submit this
                    empty_buffer = BytesIO()
                    return Response(
                        empty_buffer.getvalue(),
                        media_type="application/octet-stream",
                        status_code=503,  # Service Unavailable
                        headers={"X-TRELLIS-Error": "Service unavailable"}
                    )
                else:
                    logger.error(f"TRELLIS generation failed: {e}", exc_info=True)
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
                logger.info(f"  📊 Final stats: {timings['num_gaussians']:,} gaussians, {timings['file_size_mb']:.1f}MB, CLIP={score:.3f}")
            else:
                logger.info(f"  📊 Final stats: {timings['num_gaussians']:,} gaussians, {timings['file_size_mb']:.1f}MB (CLIP validation disabled)")
    
            # DIAGNOSTIC: Analyze PLY quality beyond just count
            try:
                from diagnostics.ply_analyzer import analyze_gaussian_quality, diagnose_ply_issues
                from diagnostics.submission_tracker import get_tracker
    
                ply_quality = analyze_gaussian_quality(ply_bytes)
                issues = diagnose_ply_issues(ply_quality)
    
                if ply_quality:
                    logger.info(f"  🔬 PLY Quality Analysis:")
                    logger.info(f"     Spatial variance: {ply_quality.get('spatial_variance', 0):.4f}")
                    logger.info(f"     Bbox volume: {ply_quality.get('bbox_volume', 0):.4f}")
                    logger.info(f"     Avg opacity: {ply_quality.get('avg_opacity', 0):.3f}")
                    logger.info(f"     Avg scale: {ply_quality.get('avg_scale', 0):.4f}")
                    logger.info(f"     Density variance: {ply_quality.get('density_variance', 0):.2f}")
    
                if issues:
                    logger.warning(f"  ⚠️  Quality issues detected:")
                    for issue in issues:
                        logger.warning(f"     {issue}")
    
                # Log submission for correlation analysis
                tracker = get_tracker()
                submission_id = tracker.log_submission(
                    prompt=prompt,
                    gaussian_count=timings['num_gaussians'],
                    file_size_mb=timings['file_size_mb'],
                    generation_time=t_total if 't_total' in locals() else (time.time() - t_start),
                    ply_quality_metrics=ply_quality,
                    clip_score_2d=flux_score if 'flux_score' in locals() else None,
                    clip_score_3d=score if 'score' in locals() else None,
                )
                logger.debug(f"  Logged submission: {submission_id}")
    
            except Exception as e:
                logger.error(f"  Diagnostic analysis failed: {e}")
    
            # Success!
            t_total = time.time() - t_start
    
            logger.info("=" * 60)
            logger.info(f"✅ GENERATION COMPLETE")
            logger.info(f"   Total time: {t_total:.2f}s")
            logger.info(f"   SDXL-Turbo (4-step): {t2-t1:.2f}s")
            logger.info(f"   Background: {t3-t2:.2f}s")
            logger.info(f"   3D (TRELLIS): {t4-t3_start:.2f}s")
            if app.state.clip_validator:
                logger.info(f"   Validation: {t6-t5:.2f}s")
                logger.info(f"   CLIP Score: {score:.3f}")
            logger.info(f"   File size: {len(ply_bytes)/1024:.1f} KB")
            logger.info("=" * 60)

            # CRITICAL: Delete large objects before returning to prevent 32GB memory accumulation
            # Without this, each generation leaves 30MB PLY + model objects in memory until GC runs
            if 'gs_model' in locals():
                del gs_model
            if 'rgba_image' in locals():
                del rgba_image
            if 'image' in locals():
                del image
            cleanup_memory()
            logger.debug("  🧹 Cleaned up large objects before response")

            return Response(
                ply_bytes,
                media_type="application/octet-stream",
                headers={
                    "X-Generation-Time": str(t_total),
                    "X-Gaussian-Count": str(timings['num_gaussians']),
                    "X-File-Size-MB": str(timings['file_size_mb']),
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
            "trellis_service": app.state.trellis_service_url,  # TRELLIS microservice URL
            # DEPRECATED: InstantMesh + Mesh-to-Gaussian (replaced by TRELLIS)
            # "instantmesh_service": app.state.instantmesh_service_url,
            # "mesh_to_gaussian": app.state.mesh_to_gaussian is not None,
            # "dreamgaussian": app.state.gaussian_models is not None,  # Commented out
            "clip_validator": app.state.clip_validator is not None
        },
        "config": {
            "flux_steps": args.flux_steps,
            "validation_enabled": args.enable_validation,
            "validation_threshold": args.validation_threshold if args.enable_validation else None,
            "scale_normalization_enabled": args.enable_scale_normalization,
            "prompt_enhancement_enabled": args.enable_prompt_enhancement,
            "image_enhancement_enabled": args.enable_image_enhancement,
            "min_gaussian_count": args.min_gaussian_count,
            "background_threshold": args.background_threshold
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
