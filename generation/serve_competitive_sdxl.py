"""
404-GEN COMPETITIVE GENERATION SERVICE (SDXL-Turbo version)

Alternative to FLUX version - uses SDXL-Turbo instead:
- No HuggingFace authentication required
- Smaller download (7GB vs 12GB)
- Still very fast and high quality

Pipeline:
1. SDXL-Turbo: Ultra-fast text-to-image (4 steps = 2-3s)
2. BRIA RMBG 2.0: SOTA background removal
3. DreamGaussian: Fast 3D generation (optimized config)
4. CLIP Validation: Pre-submission quality check

Target: <20 seconds per generation with >0.7 CLIP score
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

from omegaconf import OmegaConf

# SOTA models
from models.sdxl_turbo_generator import SDXLTurboGenerator
from models.background_remover import SOTABackgroundRemover
from validators.clip_validator import CLIPValidator

# Existing 3D generation
from DreamGaussianLib import ModelsPreLoader
from DreamGaussianLib.GaussianProcessor import GaussianProcessor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
    parser.add_argument(
        "--config",
        default="configs/text_mv_fast.yaml",
        help="DreamGaussian config (use fast config for competition)"
    )
    parser.add_argument(
        "--sdxl-steps",
        type=int,
        default=4,
        help="SDXL-Turbo inference steps (1-4 recommended, 4=best quality)"
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
        default=True,
        help="Enable CLIP validation (recommended for competition)"
    )
    return parser.parse_args()


args = get_args()
app = FastAPI(title="404-GEN Competitive Miner (SDXL-Turbo)")


# Global state
class AppState:
    """Holds all loaded models"""
    sdxl_generator: SDXLTurboGenerator = None
    background_remover: SOTABackgroundRemover = None
    gaussian_models: list = None
    clip_validator: CLIPValidator = None


app.state = AppState()


@app.on_event("startup")
def startup_event():
    """
    Load all models on startup.

    This takes ~30-60 seconds but only happens once.
    """
    logger.info("=" * 60)
    logger.info("404-GEN COMPETITIVE MINER - STARTUP (SDXL-Turbo)")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 1. Load SDXL-Turbo (text-to-image)
    logger.info("\n[1/4] Loading SDXL-Turbo (text-to-image)...")
    app.state.sdxl_generator = SDXLTurboGenerator(device=device)
    logger.info("✅ SDXL-Turbo ready")

    # 2. Load BRIA RMBG 2.0 (background removal)
    logger.info("\n[2/4] Loading BRIA RMBG 2.0 (background removal)...")
    app.state.background_remover = SOTABackgroundRemover(device=device)
    logger.info("✅ BRIA RMBG 2.0 ready")

    # 3. Load DreamGaussian models
    logger.info("\n[3/4] Loading DreamGaussian (3D generation)...")
    config = OmegaConf.load(args.config)
    app.state.gaussian_models = ModelsPreLoader.preload_model(config, device)
    logger.info("✅ DreamGaussian ready")

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
    logger.info("🚀 COMPETITIVE MINER READY FOR PRODUCTION")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"SDXL-Turbo steps: {args.sdxl_steps}")
    logger.info(f"Validation: {'ON' if args.enable_validation else 'OFF'}")
    logger.info(f"Expected speed: 15-25 seconds per generation")
    logger.info("=" * 60 + "\n")


@app.post("/generate/")
async def generate(prompt: str = Form()) -> Response:
    """
    Competitive generation pipeline.

    Pipeline:
    1. SDXL-Turbo: prompt → image (2-3s)
    2. BRIA RMBG: image → RGBA (0.2s)
    3. DreamGaussian: RGBA → Gaussian Splat (15s)
    4. CLIP Validation: quality check (0.5s)

    Total: ~20 seconds
    """
    t_start = time.time()

    # Defensive: Sanitize prompt (some miner requests send base64 images as prompts)
    original_prompt = prompt
    if len(prompt) > 200 or prompt.startswith('iVBOR') or '==' in prompt[-10:]:
        logger.warning(f"⚠️  Invalid prompt detected (length={len(prompt)}, likely base64 image)")
        logger.warning(f"   First 50 chars: {prompt[:50]}...")
        prompt = "a 3D object"  # Fallback to generic prompt
        logger.info(f"   Using fallback prompt: '{prompt}'")
    elif len(prompt) > 77:
        logger.warning(f"⚠️  Prompt too long ({len(prompt)} chars), truncating to 77")
        prompt = prompt[:77]

    logger.info(f"🎯 Generation request: '{prompt}'")

    try:
        # Step 1: Text-to-image with SDXL-Turbo
        t1 = time.time()
        logger.info("  [1/4] Generating image with SDXL-Turbo...")

        # Enhance prompt for better quality
        enhanced_prompt = f"{prompt}, highly detailed, 8k, professional 3D render, sharp focus, octane render"

        image = app.state.sdxl_generator.generate(
            prompt=enhanced_prompt,
            num_inference_steps=args.sdxl_steps,
            height=512,
            width=512
        )

        t2 = time.time()
        logger.info(f"  ✅ SDXL-Turbo done ({t2-t1:.2f}s)")

        # Free VRAM after SDXL generation (but keep image for next step)
        app.state.sdxl_generator.clear_cache()
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Step 2: Background removal with BRIA RMBG 2.0
        logger.info("  [2/4] Removing background with BRIA RMBG 2.0...")

        rgba_image = app.state.background_remover.remove_background(
            image,
            threshold=0.5
        )

        # Now free the input image
        del image
        gc.collect()
        torch.cuda.empty_cache()

        t3 = time.time()
        logger.info(f"  ✅ Background removal done ({t3-t2:.2f}s)")

        # Step 3: 3D generation with DreamGaussian
        logger.info("  [3/4] Generating 3D with DreamGaussian...")

        # Save RGBA to temp for DreamGaussian
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='_rgba.png', delete=False) as tmp:
            rgba_image.save(tmp.name)
            tmp_path = tmp.name

        # Also save prompt to caption file (DreamGaussian expects this)
        caption_path = tmp_path.replace('_rgba.png', '_caption.txt')
        with open(caption_path, 'w') as f:
            f.write(prompt)

        try:
            # Load config
            config = OmegaConf.load(args.config)
            config.input = tmp_path
            config.prompt = prompt

            # Generate
            gaussian_processor = GaussianProcessor(config, prompt)
            gaussian_processor.train(app.state.gaussian_models, config.iters)

            # Get PLY
            buffer = BytesIO()
            gaussian_processor.get_gs_model().save_ply(buffer)
            buffer.seek(0)
            ply_bytes = buffer.getvalue()

        finally:
            # Clean up temp files
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if os.path.exists(caption_path):
                os.remove(caption_path)

        t4 = time.time()
        logger.info(f"  ✅ 3D generation done ({t4-t3:.2f}s)")

        # Step 4: CLIP validation
        if app.state.clip_validator:
            logger.info("  [4/4] Validating with CLIP...")

            # Validate using the RGBA image converted to RGB (image was deleted for memory)
            validation_image = rgba_image.convert("RGB")
            passes, score = app.state.clip_validator.validate_image(
                validation_image,
                prompt
            )

            t5 = time.time()
            logger.info(f"  ✅ Validation done ({t5-t4:.2f}s)")

            if not passes:
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

            logger.info(f"  ✅ VALIDATION PASSED: CLIP={score:.3f}")

            # Clean up validation image
            del validation_image
            gc.collect()
            torch.cuda.empty_cache()

        # Success!
        t_total = time.time() - t_start

        logger.info("=" * 60)
        logger.info(f"✅ GENERATION COMPLETE")
        logger.info(f"   Total time: {t_total:.2f}s")
        logger.info(f"   SDXL-Turbo: {t2-t1:.2f}s")
        logger.info(f"   Background: {t3-t2:.2f}s")
        logger.info(f"   3D: {t4-t3:.2f}s")
        if app.state.clip_validator:
            logger.info(f"   Validation: {t5-t4:.2f}s")
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
        logger.error(f"❌ Generation failed: {e}", exc_info=True)

        # Return empty result on error (better than crash)
        empty_buffer = BytesIO()
        return Response(
            empty_buffer.getvalue(),
            media_type="application/octet-stream",
            headers={"X-Generation-Error": str(e)},
            status_code=500
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "sdxl-turbo",
        "models_loaded": {
            "sdxl_turbo": app.state.sdxl_generator is not None,
            "background_remover": app.state.background_remover is not None,
            "gaussian": app.state.gaussian_models is not None,
            "clip_validator": app.state.clip_validator is not None
        },
        "config": {
            "sdxl_steps": args.sdxl_steps,
            "validation_enabled": args.enable_validation,
            "validation_threshold": args.validation_threshold if args.enable_validation else None
        }
    }


@app.get("/stats")
async def get_stats():
    """Get generation statistics"""
    return {
        "service": "404-gen-competitive-miner-sdxl",
        "version": "1.0.0",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


if __name__ == "__main__":
    logger.info("Starting 404-GEN Competitive Generation Service (SDXL-Turbo)...")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="info"
    )
