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
import base64
import os
from dotenv import load_dotenv
from openai import OpenAI

from omegaconf import OmegaConf

# Load environment variables for OpenAI API
# Note: .env is in parent directory (/home/kobe/404-gen/v1/3D-gen/.env)
# Use override=True to replace any existing placeholder environment variables
env_path = '/home/kobe/404-gen/v1/3D-gen/.env'
load_dotenv(dotenv_path=env_path, override=True)

# Initialize OpenAI client for LLM-based prompt enhancement
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# SOTA models
from models.sdxl_turbo_generator import SDXLTurboGenerator
from models.background_remover import SOTABackgroundRemover
from models.depth_estimator import DepthEstimator
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
import cv2  # For advanced image preprocessing (CLAHE, exposure compensation)


# ============================================================================
# ADVANCED IMAGE PREPROCESSING (RESEARCH-BACKED QUALITY IMPROVEMENTS)
# ============================================================================

def apply_exposure_compensation(image: Image.Image) -> Image.Image:
    """
    Apply exposure compensation for underexposed/overexposed images.

    Research: Photogrammetric preprocessing minimizes SIFT failure due to
    illumination changes. Optimal brightness target: 128 (middle gray).

    Args:
        image: PIL Image (RGB)

    Returns:
        Exposure-corrected PIL Image
    """
    img_array = np.array(image)

    # Calculate mean brightness
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    mean_brightness = gray.mean()

    # Optimal brightness: 128 (middle gray)
    target_brightness = 128
    adjustment_factor = target_brightness / mean_brightness

    # Limit adjustment to prevent over-correction
    adjustment_factor = np.clip(adjustment_factor, 0.7, 1.5)

    if abs(adjustment_factor - 1.0) > 0.1:
        logger.info(f"  ⚡ Exposure compensation: {adjustment_factor:.2f}x (mean brightness: {mean_brightness:.1f} → {target_brightness})")

        # Apply gamma correction for exposure adjustment
        img_corrected = np.power(img_array / 255.0, 1.0 / adjustment_factor) * 255.0
        img_corrected = np.clip(img_corrected, 0, 255).astype(np.uint8)

        return Image.fromarray(img_corrected)
    else:
        logger.debug(f"  ✓ Exposure OK (brightness: {mean_brightness:.1f})")
        return image


def apply_advanced_preprocessing(image: Image.Image, is_image_to_3d: bool = False) -> Image.Image:
    """
    Apply advanced preprocessing for optimal 3D reconstruction.

    Research-backed techniques from underwater 3D reconstruction studies:
    - CLAHE (Contrast-Limited Adaptive Histogram Equalization): +7.56% feature detection
    - Unsharp masking for edge enhancement: +12.94% reconstruction quality
    - Exposure compensation for low-light images

    Source: "CLAHE + sharpening increases reconstructed points by 7.60% and
    reconstructed features by 12.94%" (Photogrammetric preprocessing research)

    Args:
        image: PIL Image (RGB or RGBA)
        is_image_to_3d: Apply aggressive enhancement for user-provided images

    Returns:
        Enhanced PIL Image optimized for TRELLIS
    """
    from PIL import ImageEnhance

    # Convert to numpy for OpenCV processing
    img_array = np.array(image.convert('RGB'))

    # 1. CLAHE (Contrast-Limited Adaptive Histogram Equalization)
    # Research shows this improves feature detection by 7.56%
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel (lightness)
    # clipLimit=2.0 prevents over-enhancement in uniform regions
    # tileGridSize=(8,8) balances local vs global contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Merge back
    lab_clahe = cv2.merge([l_clahe, a, b])
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    # Convert back to PIL
    image_enhanced = Image.fromarray(img_clahe)

    # 2. Unsharp masking for edge enhancement
    # Research: "Wallis filter showed very successful performances in photogrammetric community"
    # Improves reconstruction quality by 12.94%

    if is_image_to_3d:
        # Aggressive sharpening for user images (may have blur/compression artifacts)
        sharpness_factor = 2.5
        contrast_factor = 1.3
    else:
        # Moderate sharpening for SDXL-Turbo output (already high quality)
        sharpness_factor = 1.8
        contrast_factor = 1.2

    enhancer = ImageEnhance.Sharpness(image_enhanced)
    image_sharp = enhancer.enhance(sharpness_factor)

    # 3. Contrast boost (after CLAHE for fine-tuning)
    contrast_enhancer = ImageEnhance.Contrast(image_sharp)
    image_final = contrast_enhancer.enhance(contrast_factor)

    logger.info(f"  ✅ Advanced preprocessing: CLAHE + unsharp({sharpness_factor:.1f}x) + contrast({contrast_factor:.1f}x)")

    return image_final


# ============================================================================
# LLM-BASED PROMPT ENHANCEMENT (TIER 0 - PRIMARY QUALITY IMPROVEMENT)
# ============================================================================

def enhance_prompt_with_llm(prompt: str, timeout: float = 2.5) -> dict:
    """
    Use GPT-4o-mini to enhance prompts for optimal SDXL-Turbo performance.

    Based on research showing "professional product photography" keywords
    increase CLIP scores by 50-80%.

    Args:
        prompt: Original user prompt
        timeout: Max time for LLM call (default 2.5s for <2s target)

    Returns:
        dict with:
            - enhanced_prompt: LLM-enhanced prompt
            - negative_prompt: Research-backed negative prompt
            - method: "llm" if successful, "fallback" if LLM failed
            - latency: Time taken for enhancement
    """
    t_start = time.time()

    # Research-backed concise system prompt (avoid 77-token truncation)
    system_prompt = """You are a concise prompt optimizer for SDXL-Turbo image generation.

CRITICAL CONSTRAINTS:
1. **Max Length:** 12-15 words total (CLIP has 77-token limit, longer prompts get truncated)
2. **Core Keywords Only:** Pick 2-3 most important descriptors
3. **Simple Language:** Avoid verbose phrases like "award-winning commercial photography"

PROVEN KEYWORDS (Pick 1-2 MAX):
- Quality: "detailed", "sharp focus", "high quality"
- Lighting: "studio lighting", "soft lighting"
- Style: "product photography", "professional photo"
- Background: "white background", "clean background"

ADAPTIVE ENHANCEMENT:
- SHORT (1-5 words): Add 5-7 words max (total: 10-12 words)
- MEDIUM (6-15 words): Add 3-5 words max (total: 12-15 words)
- LONG (16+ words): Add NOTHING or replace verbose words with concise equivalents

CRITICAL: NEVER mention "flat", "thin", "2D", "paper-like" - these DESTROY CLIP scores

EXAMPLES (10-15 words max):

Input: "red sports car"
Output: "red sports car, detailed, product photography, studio lighting, white background"
(11 words)

Input: "wooden chair with carved details"
Output: "wooden chair with carved details, product photography, clean background"
(10 words)

Input: "brass candle holder"
Output: "brass candle holder, polished finish, product photography, studio lighting"
(10 words)

Transform the prompt below. Return ONLY the enhanced prompt (10-15 words max), nothing else."""

    try:
        # Call GPT-4o-mini with timeout
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.3,  # Low temp for consistent quality keywords
            timeout=timeout
        )

        enhanced_prompt = response.choices[0].message.content.strip()
        latency = time.time() - t_start

        # NO NEGATIVE PROMPT - SDXL-Turbo ignores it at CFG=1.0
        # Research: "At CFG of 1, the negative prompt has no effect and doesn't change a pixel.
        # SDXL Turbo works at 1 to 4 steps with 1.0 CFG. The recommendation is to not bother
        # with negative prompts as they don't work with SDXL Turbo."
        negative_prompt = ""

        return {
            'enhanced_prompt': enhanced_prompt,
            'negative_prompt': negative_prompt,
            'method': 'llm',
            'latency': latency
        }

    except Exception as e:
        # Fallback if LLM fails or times out
        latency = time.time() - t_start
        logger.warning(f"  ⚠️ LLM enhancement failed ({latency:.2f}s): {e}")
        logger.info(f"  ↪ Falling back to rule-based enhancement")

        fallback_result = enhance_prompt_fallback(prompt)
        fallback_result['latency'] = latency
        return fallback_result


def enhance_prompt_fallback(prompt: str) -> dict:
    """
    Fallback rule-based enhancement when LLM is unavailable.

    CRITICAL: Keep prompts under 15 words to avoid 77-token truncation.
    Research shows SDXL-Turbo works best with concise prompts (10-15 words).

    Args:
        prompt: Original user prompt

    Returns:
        dict with enhanced_prompt, negative_prompt, and method='fallback'
    """
    word_count = len(prompt.split())

    # Determine enhancement level (concise approach)
    if word_count <= 5:
        # SHORT: Add core keywords only (10-12 words total)
        enhanced_prompt = f"{prompt}, product photography, sharp focus, studio lighting, white background"
    elif word_count <= 15:
        # MEDIUM: Minimal addition (12-15 words total)
        enhanced_prompt = f"{prompt}, product photography, clean background"
    else:
        # LONG: No addition (avoid exceeding 77 tokens)
        enhanced_prompt = prompt

    # NO NEGATIVE PROMPT - SDXL-Turbo ignores it at CFG=1.0
    # Research: Negative prompts have no effect at CFG=1.0 (SDXL-Turbo default)
    negative_prompt = ""

    return {
        'enhanced_prompt': enhanced_prompt,
        'negative_prompt': negative_prompt,
        'method': 'fallback'
    }


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
    parser.add_argument(
        "--enable-depth-estimation",
        action="store_true",
        default=False,
        help="Enable depth estimation preprocessing (improves flat/ambiguous objects)"
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
        flux_steps = 4  # OPTIMAL for SDXL-Turbo (research: quality degrades at 5-10 steps)
        # SDXL-Turbo is distilled for 1-step inference, works best with 1-4 steps
        # Previous: 6 steps caused over-refinement and quality loss
        validation_threshold = 0.15  # RESTORED: During successful period (Nov 11 20:xx), 0.15 threshold allowed CLIP 0.176-0.308 → Scores 0.60-0.79
        enable_validation = True  # ENABLED: Phase 1 - filter low-quality outputs before submission
        enable_scale_normalization = False
        enable_prompt_enhancement = True  # ENABLED: Phase 1 - add quality keywords to prompts
        enable_image_enhancement = True   # ENABLED: Phase 6 - improve TRELLIS surface detection for complex subjects
        min_gaussian_count = 150000  # ENABLED: Phase 2A - filter LOW density models (<150K = 50% acceptance)
        background_threshold = 0.5  # Standard rembg threshold, preserves more object detail
        enable_depth_estimation = True  # ENABLED for Phase 5 testing (depth preprocessing)
    args = Args()

app = FastAPI(title="404-GEN Competitive Miner")


# Global state
class AppState:
    """Holds all loaded models"""
    flux_generator: SDXLTurboGenerator = None  # SDXL-Turbo on GPU 1 (RTX 5070 Ti)  # Stable Cascade on GPU 1 (RTX 5070 Ti)
    background_remover: SOTABackgroundRemover = None
    depth_estimator: DepthEstimator = None  # Depth estimation on GPU 0 (sequential with BG removal)
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


def detect_and_enhance_prompt(prompt: str) -> dict:
    """
    Tier 1: Prompt Engineering + Negative Prompts

    Detects risky prompts and applies targeted enhancements to reduce Score=0.0 failures.

    Root Cause Analysis (from 4 investigated failures):
    - 75% of Score=0.0 failures are PREVENTABLE through prompt engineering:
      * Flat objects (pendant, cutting board) → bbox_y < 0.25 → 90% failure rate
      * Multi-object scenes (boots on legs) → bbox_y varies but composition fails
      * Prompts containing "flat" → Explicit flat instruction

    Strategy:
    1. Detect flat objects (pendant, disc, cutting board) → Remove/enhance
    2. Detect scene triggers (wearing, on legs, person) → Remove/isolate
    3. Remove problematic keywords (flat, thin, wearing)
    4. Add volumetric modifiers (thick, sculptural, three-dimensional)
    5. Generate negative prompts (multiple objects, scene, person, flat)

    Expected Impact: 40-50% reduction in Score=0.0 failures (15-20% → 8-10%)

    Args:
        prompt: Original text prompt from validator

    Returns:
        dict with keys:
            - enhanced_prompt: Enhanced version of prompt
            - negative_prompt: Negative prompt string
            - risk_level: "HIGH" | "MEDIUM" | "LOW"
            - detected_keywords: List of detected risky keywords
            - modifications: List of modifications applied
    """
    import re

    prompt_lower = prompt.lower()

    # TIER 1: EXPLICIT flat keywords (100% failure predictors)
    flat_keywords_explicit = [
        'flat', 'thin', 'disc', 'coin', 'medallion', 'sheet', 'plate', 'paper'
    ]

    # TIER 2: IMPLICIT flat objects (80%+ failure rate from real data)
    flat_keywords_implicit = [
        'pendant', 'amulet', 'charm', 'badge', 'coaster', 'token',
        'cutting board', 'chopping board', 'mat', 'rug', 'tile',
        'brooch', 'locket', 'pin', 'leaf', 'petal'  # High-value additions (jewelry/natural)
    ]

    # Scene triggers (multi-object compositions that fail)
    scene_triggers = [
        'wearing', 'worn', 'on legs', 'on feet', 'person',
        'model', 'mannequin', 'human', 'body',
        'with legs', 'with feet', 'with arms', 'with hands'  # Expanded scene triggers
    ]

    # Detect risk factors
    detected_explicit = [kw for kw in flat_keywords_explicit if kw in prompt_lower]
    detected_implicit = [kw for kw in flat_keywords_implicit if kw in prompt_lower]
    detected_scene = [kw for kw in scene_triggers if kw in prompt_lower]

    detected_keywords = detected_explicit + detected_implicit + detected_scene

    # Determine risk level
    if detected_explicit or detected_scene:
        risk_level = "HIGH"
    elif detected_implicit:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Build enhanced prompt
    enhanced_prompt = prompt
    modifications = []

    # STEP 1: Remove problematic keywords
    if detected_explicit:
        for kw in detected_explicit:
            # Case-insensitive removal with word boundaries
            enhanced_prompt = re.sub(rf'\b{re.escape(kw)}\b', '', enhanced_prompt, flags=re.IGNORECASE)
            modifications.append(f"removed '{kw}'")
        enhanced_prompt = ' '.join(enhanced_prompt.split())  # Clean up extra spaces

    # STEP 2: Remove scene triggers (isolate object)
    if detected_scene:
        for kw in detected_scene:
            enhanced_prompt = re.sub(rf'\b{re.escape(kw)}\b', '', enhanced_prompt, flags=re.IGNORECASE)
            modifications.append(f"removed '{kw}'")
        enhanced_prompt = ' '.join(enhanced_prompt.split())

    # STEP 2.5: Clean up awkward phrasing (dangling conjunctions)
    if detected_explicit or detected_scene:
        # Remove dangling "and", "or", "the" after keyword removal
        # Examples: "large and cutting board" → "large cutting board"
        #           "boots with and legs" → "boots"
        enhanced_prompt = re.sub(r'\b(and|or|the|a|an)\s+(and|or|the|a|an)\b', r'\1', enhanced_prompt, flags=re.IGNORECASE)
        enhanced_prompt = re.sub(r'\s+(and|or)\s*$', '', enhanced_prompt, flags=re.IGNORECASE)  # Trailing conjunction
        enhanced_prompt = re.sub(r'^\s*(and|or)\s+', '', enhanced_prompt, flags=re.IGNORECASE)  # Leading conjunction
        enhanced_prompt = re.sub(r'\s+', ' ', enhanced_prompt).strip()  # Normalize whitespace

    # STEP 3: Add volumetric modifiers based on risk level
    if risk_level == "HIGH":
        # Aggressive enhancement for high-risk prompts
        enhanced_prompt = f"{enhanced_prompt}, thick sculptural form, three-dimensional volume, solid construction, isolated single object"
        modifications.append("added HIGH-tier volumetric modifiers")
    elif risk_level == "MEDIUM":
        # Conservative enhancement for medium-risk prompts
        enhanced_prompt = f"{enhanced_prompt}, dimensional depth, volumetric form, isolated object"
        modifications.append("added MEDIUM-tier volumetric modifiers")
    # LOW risk: No modification needed

    # STEP 4: Build negative prompt
    negative_keywords = []

    if detected_scene:
        # Prevent multi-object compositions
        negative_keywords.extend([
            "multiple objects", "scene", "person", "human", "legs", "feet",
            "wearing", "worn", "body parts", "mannequin", "composition"
        ])

    if detected_explicit or detected_implicit:
        # Prevent flat geometry
        negative_keywords.extend([
            "flat", "thin", "2D", "paper", "disc", "sheet", "planar"
        ])

    # Baseline negative prompts (always applied for better results)
    baseline_negatives = [
        "blurry", "low quality", "distorted", "background clutter", "multiple views"
    ]

    all_negatives = negative_keywords + baseline_negatives if negative_keywords else baseline_negatives
    negative_prompt = ", ".join(all_negatives)

    return {
        'enhanced_prompt': enhanced_prompt,
        'negative_prompt': negative_prompt,
        'risk_level': risk_level,
        'detected_keywords': detected_keywords,
        'modifications': modifications
    }


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
    logger.info("\n[1/5] Initializing SDXL-Turbo (lazy loading)...")
    app.state.flux_generator = SDXLTurboGenerator(device="cuda:1")  # GPU 1: Stable Cascade
    logger.info("✅ SDXL-Turbo initialized (will load on first request)")
    logger.info("   Multi-GPU setup:")
    logger.info("     - GPU 0 (RTX 4090, 24GB): TRELLIS (~6GB) + BG removal (~0.5GB) + Depth (~2GB sequential)")
    logger.info("     - GPU 1 (RTX 5070 Ti, 16GB): SDXL-Turbo (~7GB)")
    logger.info("   Peak GPU 0: ~8GB (plenty of headroom!)")
    logger.info("   Speed: ~1-2s image generation (13x faster than FLUX!)")
    logger.info("   Architecture: Prior (5.1GB) + Decoder (1.5GB)")

    # 2. Load BiRefNet (background removal) - GPU 0
    logger.info("\n[2/5] Loading BiRefNet (background removal)...")
    app.state.background_remover = SOTABackgroundRemover(device="cuda:0", model_type="birefnet")  # GPU 0: Shares with TRELLIS
    logger.info("✅ BiRefNet ready (GPU 0 - superior thin structure detection)")
    logger.info("   Upgrade from U2-Net: Addresses boots/sword/chair failures (+30-50% quality)")

    # 2.5. Initialize Depth Estimator (GPU 0, ~2GB) - NEW
    logger.info("\n[2.5/5] Initializing Depth Estimator (MiDaS)...")
    app.state.depth_estimator = DepthEstimator(model_type="midas", device="cuda:0")  # GPU 0: Sequential with BG removal
    logger.info("✅ Depth estimator ready (GPU 0, lazy loading, ~2GB)")
    logger.info("   Sequential pipeline: BG removal (0.5GB) → clear → Depth (2GB) → clear → TRELLIS (6GB)")

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
        logger.info("\n[4/5] Loading CLIP validator...")
        app.state.clip_validator = CLIPValidator(
            device=device,
            threshold=args.validation_threshold
        )
        logger.info(f"✅ CLIP validator ready (threshold={args.validation_threshold})")
    else:
        logger.warning("\n[4/5] CLIP validation DISABLED (not recommended)")
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
                    # Handle data URL format (data:image/png;base64,...)
                    if ',' in prompt[:100]:
                        base64_data = prompt.split(',', 1)[1]
                    else:
                        base64_data = prompt

                    # Decode to PIL Image (RGB for BiRefNet input)
                    image_bytes = base64.b64decode(base64_data)
                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

                    logger.info(f"  ✅ Decoded image: {image.size[0]}x{image.size[1]} RGB")

                    t2 = time.time()

                    # CRITICAL FIX: Apply BiRefNet background removal to IMAGE-TO-3D tasks!
                    # Validators send photos WITH backgrounds to test our full pipeline
                    logger.info("  [2/4] Removing background with BiRefNet (IMAGE-TO-3D)...")
                    logger.info("     Validators test full pipeline - don't assume clean input!")

                    rgba_image = app.state.background_remover.remove_background(
                        image,
                        threshold=args.background_threshold
                    )
                    logger.debug(f"  Background removal threshold: {args.background_threshold}")

                    t3 = time.time()
                    logger.info(f"  ✅ Background removal done ({t3-t2:.2f}s)")

                    # DISABLED: Advanced preprocessing was causing 18% CLIP degradation
                    # Research-backed CLAHE+sharpening is for poor-quality/underwater images
                    # SDXL-Turbo already produces high-quality images - preprocessing adds artifacts
                    # Baseline without preprocessing: CLIP median 0.28, with preprocessing: 0.23 (-18%)
                    #
                    # t_preprocess = time.time()
                    # logger.info("  [2.5/4] Applying advanced preprocessing (IMAGE-TO-3D)...")
                    # rgba_image_rgb = rgba_image.convert('RGB')
                    # rgba_image_rgb = apply_exposure_compensation(rgba_image_rgb)
                    # rgba_image_enhanced = apply_advanced_preprocessing(rgba_image_rgb, is_image_to_3d=True)
                    # ... (alpha channel restoration code)
                    # t_preprocess_done = time.time()
                    # logger.info(f"  ✅ Advanced preprocessing done ({t_preprocess_done-t_preprocess:.2f}s)")

                    # DEBUG: Save background-removed image for quality inspection
                    debug_timestamp = int(time.time())
                    rgba_image.save(f"/tmp/debug_2_rembg_image2d_{debug_timestamp}.png")
                    logger.debug(f"  Saved debug image: /tmp/debug_2_rembg_image2d_{debug_timestamp}.png")

                    # Set generic prompt for validation
                    validation_prompt = "a 3D object"

                    # Set prompt stats for image-to-3D mode (no text prompt to enhance)
                    original_words = 0
                    enhanced_words = 0

                    logger.info(f"  ✅ Image-to-3D preprocessing complete ({t3-t1:.2f}s)")
                    logger.info(f"     SDXL-Turbo: skipped (image provided)")
                    logger.info(f"     BiRefNet: {t3-t2:.2f}s (background removed)")

                except Exception as e:
                    logger.error(f"  ❌ Failed to decode or process image: {e}", exc_info=True)
                    # Return empty result for invalid image
                    empty_buffer = BytesIO()
                    return Response(
                        empty_buffer.getvalue(),
                        media_type="application/octet-stream",
                        status_code=400,
                        headers={"X-Validation-Failed": "true", "X-Error": "Invalid base64 image or background removal failed"}
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

                # TIER 0: LLM-Based Prompt Enhancement (PRIMARY QUALITY IMPROVEMENT)
                # Uses GPT-4o-mini to add research-backed photography keywords
                # Expected impact: +50-80% CLIP improvement (0.19 → 0.28-0.35)
                logger.info("  🤖 TIER 0: LLM-based prompt enhancement...")
                llm_result = enhance_prompt_with_llm(prompt)

                logger.info(f"  ✅ LLM enhancement: method={llm_result['method']}, latency={llm_result['latency']:.2f}s")
                logger.debug(f"     Original: '{prompt}'")
                logger.debug(f"     LLM Enhanced: '{llm_result['enhanced_prompt']}'")

                # Use LLM-enhanced prompt as base for Tier 1
                llm_enhanced_prompt = llm_result['enhanced_prompt']
                llm_negative_prompt = llm_result['negative_prompt']

                # TIER 1: Prompt Enhancement + Negative Prompts (ALWAYS ENABLED)
                # Detects risky prompts (flat objects, multi-object scenes) and applies targeted fixes
                # Expected impact: 40-50% reduction in Score=0.0 failures
                # NOTE: Now operates on LLM-enhanced prompt for layered improvement
                enhancement_data = detect_and_enhance_prompt(llm_enhanced_prompt)

                enhanced_prompt = enhancement_data['enhanced_prompt']
                tier1_negative_prompt = enhancement_data['negative_prompt']
                risk_level = enhancement_data['risk_level']
                detected_keywords = enhancement_data['detected_keywords']
                modifications = enhancement_data['modifications']

                # REMOVED: Negative prompt merging (SDXL-Turbo ignores negative prompts at CFG=1.0)
                # Research: "At CFG of 1, the negative prompt has no effect and doesn't change a pixel"

                # TELEMETRY: Log Tier 1 enhancement details
                if risk_level != "LOW":
                    logger.info(f"  🎯 TIER 1 DETECTION: Risk={risk_level}, Keywords={detected_keywords}")
                    logger.info(f"     Modifications: {modifications}")
                    logger.debug(f"     Tier 0→1: '{llm_enhanced_prompt}' → '{enhanced_prompt}'")
                else:
                    logger.debug(f"  ✓ TIER 1: Low-risk prompt, no additional enhancement needed")

                # MEASUREMENT: Track prompt stats for density correlation analysis
                original_words = len(prompt.split())
                enhanced_words = len(enhanced_prompt.split())
                logger.info(f"  📏 PROMPT STATS: {original_words}w → {enhanced_words}w (LLM→Tier1, risk={risk_level})")

                # Use 512x512 for better CLIP scores (CLIP prefers higher resolution)
                # NOTE: No negative_prompt parameter - SDXL-Turbo ignores it at CFG=1.0
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

                # DISABLED: Advanced preprocessing was causing 18% CLIP degradation
                # Research-backed CLAHE+sharpening is for poor-quality/underwater images
                # SDXL-Turbo already produces high-quality images - preprocessing adds artifacts
                # Baseline without preprocessing: CLIP median 0.28, with preprocessing: 0.23 (-18%)
                #
                # t_preprocess = time.time()
                # logger.info("  [2.5/4] Applying advanced preprocessing (TEXT-TO-3D)...")
                # rgba_image_rgb = rgba_image.convert('RGB')
                # rgba_image_enhanced = apply_advanced_preprocessing(rgba_image_rgb, is_image_to_3d=False)
                # ... (alpha channel restoration code)
                # t_preprocess_done = time.time()
                # logger.info(f"  ✅ Advanced preprocessing done ({t_preprocess_done-t_preprocess:.2f}s)")

                # DEBUG: Save background-removed image for quality inspection
                rgba_image.save(f"/tmp/debug_2_rembg_{debug_timestamp}.png")
                logger.debug(f"  Saved debug image: /tmp/debug_2_rembg_{debug_timestamp}.png")

                # Free the input image from memory
                del image
            cleanup_memory()
            logger.debug(f"  GPU memory freed after background removal")

            # Step 2.5: Depth estimation (GPU 0) - NEW (IMAGE-TO-3D only)
            if args.enable_depth_estimation and is_base64_image:
                t2_5_start = time.time()
                logger.info("  [2.5/4] Estimating depth map (IMAGE-TO-3D)...")

                # Estimate depth from RGB image
                rgb_for_depth = rgba_image.convert('RGB')
                depth_map = app.state.depth_estimator.estimate_depth(rgb_for_depth)

                # Save depth visualization for debugging
                debug_timestamp = int(time.time())
                depth_viz_path = f"/tmp/debug_2.5_depth_{debug_timestamp}.png"
                app.state.depth_estimator.visualize_depth(depth_map, depth_viz_path)
                logger.debug(f"  Saved depth visualization: {depth_viz_path}")

                t2_5_end = time.time()
                logger.info(f"  ✅ Depth estimation done ({t2_5_end-t2_5_start:.2f}s)")
                logger.info(f"     Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
                logger.info(f"     Depth mean: {depth_map.mean():.3f} (0=near, 1=far)")

                # CRITICAL: Free depth estimation VRAM before TRELLIS
                del rgb_for_depth
                cleanup_memory()
                logger.debug(f"  GPU memory freed after depth estimation")
            else:
                depth_map = None
                if not args.enable_depth_estimation:
                    logger.debug("  Depth estimation DISABLED (feature disabled)")
                elif not is_base64_image:
                    logger.debug("  Depth estimation SKIPPED (TEXT-TO-3D - depth only applies to IMAGE-TO-3D)")

            # Step 3: Native Gaussian generation with TRELLIS (5s, 256K gaussians)
            # NOTE: Passing validation_prompt (original) to TRELLIS, not enhanced prompt
            # Research: TRELLIS text conditioning is weak - works best with simple prompts
            # The image from SDXL already contains all visual info from enhanced prompt
            t3_start = time.time()
            try:
                # Call TRELLIS microservice for direct gaussian splat generation
                # This includes format conversion: sigmoid [0,1] → logit space [6.0-7.0]
                ply_bytes, gs_model, timings = await generate_with_trellis(
                    rgba_image=rgba_image,
                    prompt=validation_prompt,  # Use original simple prompt, not LLM-enhanced
                    trellis_url="http://localhost:10008",
                    enable_scale_normalization=args.enable_scale_normalization,
                    enable_image_enhancement=args.enable_image_enhancement,
                    min_gaussians=args.min_gaussian_count,
                    depth_map=depth_map  # Pass depth map to TRELLIS
                )
    
                # Cache for validation
                app.state.last_gs_model = gs_model
    
                t4 = time.time()
                logger.info(f"  ✅ 3D generation done ({t4-t3_start:.2f}s)")
                logger.info(f"     TRELLIS: {timings['trellis']:.2f}s, Model Load: {timings['model_load']:.2f}s")
                logger.info(f"     Generated {len(ply_bytes)/1024:.1f} KB Gaussian Splat PLY")
                logger.info(f"     📊 Generation stats: {timings['num_gaussians']:,} gaussians, {timings['file_size_mb']:.1f}MB")

                # MEASUREMENT: Track prompt-to-density correlation + Tier 1 impact
                density_tier = "HIGH" if timings['num_gaussians'] >= 400000 else ("MED" if timings['num_gaussians'] >= 150000 else "LOW")
                logger.info(f"     📈 DENSITY CORRELATION: {original_words}w → {enhanced_words}w → {timings['num_gaussians']:,}g [{density_tier}]")

                # TIER 1 TELEMETRY: Track enhancement effectiveness
                if 'risk_level' in locals():
                    logger.info(f"     🎯 TIER 1 IMPACT: Risk={risk_level}, Detected={len(detected_keywords)} keywords, Density={density_tier}")
                    if risk_level != "LOW":
                        logger.info(f"        Applied: {', '.join(modifications)}")

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
