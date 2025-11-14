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
# NEW: LLaVA for IMAGE-TO-3D text conditioning
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
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

# Data logger for comprehensive production tracking
from data_logger import init_logger, get_logger, log_startup_config
import traceback

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
        validation_threshold = 0.10  # PHASE 1.2: Lowered from 0.15 - more permissive to reduce false-positive rejections
        enable_validation = True  # ENABLED: Phase 1 - filter low-quality outputs before submission
        enable_scale_normalization = False
        enable_prompt_enhancement = True  # ENABLED: Phase 1 - add quality keywords to prompts
        enable_image_enhancement = True   # ENABLED: Phase 6 - improve TRELLIS surface detection for complex subjects
        min_gaussian_count = 50000  # PHASE 1.1: Lowered from 150K - let validators judge quality (simple objects naturally produce 60K-120K)
        background_threshold = 0.4  # PHASE 2.3: Lowered from 0.5 - preserve thin structures (chair legs, handles)
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
    llava_tokenizer = None  # NEW: LLaVA for IMAGE-TO-3D text conditioning
    llava_model = None      # NEW: LLaVA for IMAGE-TO-3D text conditioning
    llava_image_processor = None  # NEW: LLaVA image processor
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


def generate_llava_caption(image: Image.Image) -> str:
    """
    Generate detailed caption using LLaVA for IMAGE-TO-3D text conditioning.

    Research shows LLaVA provides more detailed spatial descriptions than BLIP2,
    which is critical for accurate 3D reconstruction. LLaVA describes shapes,
    materials, spatial relationships with higher fidelity.

    Args:
        image: PIL Image (RGB)

    Returns:
        Detailed caption describing object's 3D structure
    """
    if app.state.llava_model is None:
        logger.warning("  LLaVA not loaded, using generic caption")
        return "a 3D object"

    try:
        # Prepare prompt for 3D-focused description
        prompt = "Describe this object in detail, focusing on its shape, materials, and 3D structure. Be specific about dimensions, parts, and spatial relationships."

        # Format for LLaVA
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        prompt_formatted = conv.get_prompt()

        # Process image
        image_tensor = process_images([image], app.state.llava_image_processor, app.state.llava_model.config)
        image_tensor = [img.to(dtype=torch.float16, device="cuda:0") for img in image_tensor]

        # Tokenize
        input_ids = tokenizer_image_token(
            prompt_formatted,
            app.state.llava_tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).to("cuda:0")

        # Generate caption
        with torch.inference_mode():
            output_ids = app.state.llava_model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=False,
                max_new_tokens=128,
                use_cache=True,
            )

        # Decode
        caption = app.state.llava_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return caption if caption else "a 3D object"

    except Exception as e:
        logger.error(f"  LLaVA caption generation failed: {e}")
        return "a 3D object"


def smart_background_removal_with_context(
    image: Image.Image,
    background_remover: SOTABackgroundRemover,
    threshold: float = 0.3
) -> Image.Image:
    """
    Advanced background removal that preserves depth cues.

    Unlike aggressive removal, this keeps:
    - Shadows (indicate lighting direction and object height)
    - Reflections (show material properties)
    - Ground plane (provides scale reference)
    - Edge context (helps depth estimation)

    Research: Depth cues are critical for accurate 3D reconstruction.
    Aggressive background removal destroys these cues → poor geometry.

    Args:
        image: Input RGB image
        background_remover: BiRefNet instance
        threshold: Segmentation threshold (lower = more context preserved)

    Returns:
        RGBA image with smart background removal
    """
    import cv2

    logger.debug("  Applying smart background removal (preserves depth cues)...")

    # 1. Get BiRefNet mask (but use low threshold)
    rgba_image = background_remover.remove_background(image, threshold=threshold)
    mask = np.array(rgba_image.split()[3])  # Extract alpha channel

    # 2. Dilate mask to include shadows and reflections
    # Research: Shadows are critical depth cues for 3D reconstruction
    kernel_size = 21  # Larger = more context preserved
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_dilated = cv2.dilate(mask, kernel, iterations=2)

    # 3. Preserve ground plane (bottom 20% of image)
    # Research: Ground plane provides scale reference for TRELLIS
    h, w = mask.shape
    ground_plane_start = int(h * 0.80)
    mask_dilated[ground_plane_start:, :] = 255  # Keep entire bottom region

    logger.debug(f"    Preserved ground plane: bottom {100-80}% of image")

    # 4. Apply feathered mask (smooth edges)
    # Soft edges look more natural and preserve boundary detail
    mask_feathered = cv2.GaussianBlur(mask_dilated, (31, 31), 0)
    mask_feathered = mask_feathered.astype(float) / 255.0

    # 5. Blend with white background (NOT transparent)
    # Research: White background better than transparent for 3D models
    img_array = np.array(image.convert('RGB')).astype(float)
    white_background = np.ones_like(img_array) * 255

    # Blend: object (mask=1) to white (mask=0)
    result_array = img_array * mask_feathered[:, :, np.newaxis] + \
                   white_background * (1 - mask_feathered[:, :, np.newaxis])

    result_rgb = Image.fromarray(result_array.astype(np.uint8))

    # Convert back to RGBA for consistency
    result_rgba = Image.new('RGBA', result_rgb.size)
    result_rgba.paste(result_rgb, (0, 0))
    result_rgba.putalpha(Image.fromarray((mask_feathered * 255).astype(np.uint8)))

    logger.debug("  ✅ Smart background removal complete (shadows + ground plane preserved)")

    return result_rgba


def estimate_prompt_success_probability(prompt: str) -> dict:
    """
    Predict likelihood of validator acceptance based on object type.

    Data from 200 mainnet generations (Nov 13, 2025):
    - Jewelry: 100% success (13/13)
    - Animals: 100% success (5/5)
    - Furniture: 100% success (5/5)
    - Kitchenware: 40% success (6/15) ❌

    Research: Simple, solid objects succeed. Thin/concave objects fail.
    Kitchenware (bowls, cups) have concave surfaces that confuse 3D reconstruction.

    Args:
        prompt: Text prompt from validator

    Returns:
        dict with 'probability' (0.0-1.0) and 'reason' (explanation)
    """
    prompt_lower = prompt.lower()

    # Category definitions
    HIGH_SUCCESS = {
        'jewelry': ['ring', 'necklace', 'earring', 'bracelet', 'pendant', 'brooch', 'charm'],
        'animals': ['dog', 'cat', 'bird', 'horse', 'rabbit', 'bear', 'deer', 'lion'],
        'furniture': ['chair', 'table', 'desk', 'sofa', 'couch', 'bench', 'stool', 'cabinet']
    }

    LOW_SUCCESS = {
        'kitchenware': ['bowl', 'cup', 'mug', 'plate', 'dish', 'glass', 'saucer'],
        'thin_objects': ['disc', 'coin', 'medallion', 'sheet', 'paper'],
        'complex_multi': ['scene', 'room', 'landscape', 'multiple']
    }

    # Check for high-success categories
    for category, keywords in HIGH_SUCCESS.items():
        if any(kw in prompt_lower for kw in keywords):
            return {
                'probability': 1.0,
                'reason': f'High-success category: {category} (100% historical success)'
            }

    # Check for low-success categories
    for category, keywords in LOW_SUCCESS.items():
        if any(kw in prompt_lower for kw in keywords):
            # Exception: Decorative kitchenware is okay
            if category == 'kitchenware' and any(word in prompt_lower for word in ['decorative', 'ornate', 'carved', 'crystal', 'antique']):
                return {
                    'probability': 0.8,
                    'reason': f'Decorative {category} (exception to low-success rule)'
                }

            success_rates = {
                'kitchenware': 0.40,
                'thin_objects': 0.30,
                'complex_multi': 0.50
            }

            return {
                'probability': success_rates[category],
                'reason': f'Low-success category: {category} ({success_rates[category]*100:.0f}% historical success)'
            }

    # Default: neutral (no clear indicators)
    return {
        'probability': 0.75,
        'reason': 'Neutral category (average success rate)'
    }


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

    # 4.5. Load LLaVA-1.5 for IMAGE-TO-3D text conditioning
    logger.info("\n[4.5/5] Loading LLaVA-1.5 for IMAGE-TO-3D text conditioning...")
    try:
        model_path = "liuhaotian/llava-v1.5-7b"

        # Load LLaVA model components with INT8 quantization to save memory
        # FP16: 14.6GB, INT8: ~7-8GB (frees 6-7GB for TRELLIS)
        app.state.llava_tokenizer, app.state.llava_model, app.state.llava_image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device_map="cuda:0",  # Same GPU as TRELLIS/BiRefNet
            load_8bit=True  # INT8 quantization to reduce memory from 14.6GB to ~7GB
        )

        logger.info("✅ LLaVA-1.5 ready (7B model, GPU 0, INT8 quantized)")
        logger.info("   Memory: ~7-8GB (INT8 quantization, down from 14.6GB FP16)")
        logger.info("   Purpose: Generate detailed spatial captions for IMAGE-TO-3D tasks")
        logger.info("   Expected: +28% CLIP improvement (vs BLIP2's +20%)")
        logger.info("   Research: LLaVA provides detailed spatial descriptions critical for 3D")
    except Exception as e:
        logger.error(f"⚠️ LLaVA loading failed: {e}")
        logger.warning("   IMAGE-TO-3D will continue without text conditioning")
        app.state.llava_model = None
        app.state.llava_tokenizer = None
        app.state.llava_image_processor = None

    # 5. Initialize Generation Data Logger
    logger.info("\n[5/5] Initializing Generation Data Logger...")
    try:
        data_logger = init_logger(
            data_dir="/home/kobe/404-gen/v1/3D-gen/data",
            miner_uid=102,  # Current mainnet UID
            miner_version="competitive_v1.0_mainnet",
            network="mainnet",
            store_images=True,
            store_ply_files=True,
            ply_min_score_threshold=0.7,  # Only store high-quality PLYs (≥0.7)
            store_rejected_sample_rate=0.1,  # 10% of rejected for debugging
        )
        app.state.data_logger = data_logger
        logger.info("✅ Data logger ready")
        logger.info(f"   Logging to: /home/kobe/404-gen/v1/3D-gen/data/generation_history.jsonl")
        logger.info(f"   Images: /home/kobe/404-gen/v1/3D-gen/data/images/")
        logger.info(f"   PLY files (score ≥0.7): /home/kobe/404-gen/v1/3D-gen/data/ply_files/")

        # Log startup configuration
        startup_config = {
            "sdxl_turbo_steps": args.flux_steps,
            "trellis_sparse_steps": 45,  # From serve_trellis.py
            "trellis_slat_steps": 35,
            "trellis_sparse_cfg": 9.0,
            "trellis_slat_cfg": 4.0,
            "clip_threshold": args.validation_threshold if args.enable_validation else None,
            "background_threshold": args.background_threshold,
            "validation_enabled": args.enable_validation,
            "prompt_enhancement_enabled": args.enable_prompt_enhancement,
        }
        log_startup_config(startup_config, "Competitive miner startup")

    except Exception as e:
        logger.error(f"⚠️ Data logger initialization failed (non-fatal): {e}")
        logger.error(traceback.format_exc())
        app.state.data_logger = None

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

        # QUALITY GATE: Skip low-probability prompts (TEXT-TO-3D only)
        # Data from 200 mainnet generations shows kitchenware has only 40% success rate
        success_prediction = estimate_prompt_success_probability(prompt)

        logger.info(f"  🎯 Success prediction: {success_prediction['probability']*100:.0f}%")
        logger.debug(f"     Reason: {success_prediction['reason']}")

        # Skip if predicted success < 50%
        if success_prediction['probability'] < 0.5:
            logger.warning(f"  ⚠️  LOW SUCCESS PROBABILITY: {success_prediction['probability']*100:.0f}%")
            logger.warning(f"     Reason: {success_prediction['reason']}")
            logger.warning(f"     Skipping to avoid wasted compute")

            # Return empty result with skip reason
            empty_buffer = BytesIO()
            return Response(
                empty_buffer.getvalue(),
                media_type="application/octet-stream",
                status_code=200,
                headers={"X-Skip-Reason": f"Low success probability: {success_prediction['reason']}"}
            )

    # Start data logging
    log_id = None
    if hasattr(app.state, 'data_logger') and app.state.data_logger:
        try:
            task_type = "IMAGE-TO-3D" if is_base64_image else "TEXT-TO-3D"

            # Get current miner config
            miner_config = {
                "sdxl_turbo_steps": args.flux_steps,
                "trellis_sparse_steps": 45,
                "trellis_slat_steps": 35,
                "trellis_sparse_cfg": 9.0,
                "trellis_slat_cfg": 4.0,
                "clip_threshold": args.validation_threshold if args.enable_validation else None,
                "background_threshold": args.background_threshold,
                "validation_enabled": args.enable_validation,
            }

            # Start logging (non-blocking)
            log_id = app.state.data_logger.start_generation(
                task_type=task_type,
                prompt=prompt,  # Will be hashed/truncated if base64
                validator_uid=None,  # Not available at worker level
                validator_hotkey=None,
                miner_config=miner_config
            )
            logger.debug(f"📊 Started logging generation: {log_id}")
        except Exception as e:
            logger.error(f"Data logger error (non-fatal): {e}")
            log_id = None

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

                    # CRITICAL FIX: Apply smart background removal to IMAGE-TO-3D tasks!
                    # Validators send photos WITH backgrounds to test our full pipeline
                    logger.info("  [2/4] Smart background removal with BiRefNet (IMAGE-TO-3D)...")
                    logger.info("     Preserving shadows, reflections, ground plane for better 3D...")

                    rgba_image = smart_background_removal_with_context(
                        image,
                        app.state.background_remover,
                        threshold=0.3  # Lower threshold = more context preserved
                    )
                    logger.debug("  Background removal mode: smart (preserves depth cues)")

                    t3 = time.time()
                    logger.info(f"  ✅ Smart background removal done ({t3-t2:.2f}s)")

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

                    # NEW: Generate detailed caption using LLaVA
                    logger.info("  [2.5/4] Generating caption with LLaVA...")
                    t_caption_start = time.time()

                    caption = generate_llava_caption(rgba_image.convert('RGB'))
                    t_caption_end = time.time()

                    logger.info(f"  ✅ LLaVA caption generated ({t_caption_end - t_caption_start:.2f}s)")
                    logger.info(f"     Caption: \"{caption}\"")

                    # Use caption for TRELLIS text conditioning
                    enhanced_prompt = f"{caption}, 3D object, product photography"
                    validation_prompt = caption  # Use caption for CLIP validation too

                    # Set prompt stats for image-to-3D mode
                    original_words = 0
                    enhanced_words = len(caption.split())

                    logger.info(f"  ✅ Image-to-3D preprocessing complete ({t3-t1:.2f}s)")
                    logger.info(f"     SDXL-Turbo: skipped (image provided)")
                    logger.info(f"     BiRefNet: {t3-t2:.2f}s (smart mode, depth cues preserved)")
                    logger.info(f"     LLaVA: {t_caption_end - t_caption_start:.2f}s (detailed caption generated)")

                    # Log caption timing
                    if log_id and hasattr(app.state, 'data_logger') and app.state.data_logger:
                        try:
                            app.state.data_logger.log_timing(log_id, "llava_caption_time", t_caption_end - t_caption_start)
                        except Exception as e:
                            logger.debug(f"Data logger timing error (non-fatal): {e}")

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

                # Log enhanced prompt to data logger
                if log_id and hasattr(app.state, 'data_logger') and app.state.data_logger:
                    try:
                        app.state.data_logger.log_enhanced_prompt(log_id, enhanced_prompt, llm_negative_prompt)
                    except Exception as e:
                        logger.debug(f"Data logger enhanced_prompt error (non-fatal): {e}")

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

                # Log timing
                if log_id and hasattr(app.state, 'data_logger') and app.state.data_logger:
                    try:
                        app.state.data_logger.log_timing(log_id, "sdxl_time", t2-t1)
                    except Exception as e:
                        logger.debug(f"Data logger timing error (non-fatal): {e}")

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

                # Log timing
                if log_id and hasattr(app.state, 'data_logger') and app.state.data_logger:
                    try:
                        app.state.data_logger.log_timing(log_id, "background_removal_time", t3-t2)
                    except Exception as e:
                        logger.debug(f"Data logger timing error (non-fatal): {e}")

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
            # REVERTED: Priority 6 was causing high rejections (37.5% acceptance)
            # Using enhanced_prompt for better 3D guidance - text conditioning is weak but NOT harmful
            t3_start = time.time()
            try:
                # Call TRELLIS microservice for direct gaussian splat generation
                # This includes format conversion: sigmoid [0,1] → logit space [6.0-7.0]
                ply_bytes, gs_model, timings = await generate_with_trellis(
                    rgba_image=rgba_image,
                    prompt=enhanced_prompt,  # Use enhanced prompt for better 3D guidance
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

                # Log timing and output
                if log_id and hasattr(app.state, 'data_logger') and app.state.data_logger:
                    try:
                        app.state.data_logger.log_timing(log_id, "trellis_time", t4-t3_start)
                        # Also log output metrics (will be updated with CLIP score later)
                        app.state.data_logger.log_output(
                            log_id,
                            num_gaussians=timings['num_gaussians'],
                            file_size_mb=timings['file_size_mb'],
                        )
                    except Exception as e:
                        logger.debug(f"Data logger error (non-fatal): {e}")

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

                            # MULTI-VIEW CLIP: Evaluate all views and use the BEST one
                            # This prevents using a bad camera angle (e.g., back of object)
                            app.state.clip_validator.to_gpu()
                            view_scores = []
                            for i, view in enumerate(rendered_views):
                                _, view_score = app.state.clip_validator.validate_image(view, validation_prompt)
                                view_scores.append(view_score)
                            app.state.clip_validator.to_cpu()

                            # Use the view with highest CLIP score
                            best_view_idx = view_scores.index(max(view_scores))
                            validation_image = rendered_views[best_view_idx]
                            logger.info(f"  📊 Multi-view CLIP scores: {[f'{s:.3f}' for s in view_scores]}")
                            logger.info(f"  ✅ Selected view {best_view_idx+1}/{len(rendered_views)} (score={view_scores[best_view_idx]:.3f})")

                            # CRITICAL: Check view consistency (prevent inconsistent geometry submission)
                            view_variance = max(view_scores) - min(view_scores)
                            min_view_score = min(view_scores)
                            max_view_score = max(view_scores)

                            logger.info(f"  📊 View consistency: variance={view_variance:.3f}, min={min_view_score:.3f}, max={max_view_score:.3f}")

                            # Quality gate: Reject if ANY view is poor (validators render from random angles)
                            VARIANCE_THRESHOLD = 0.08  # Models with >0.08 variance get Score=0.0
                            MIN_ACCEPTABLE_VIEW = 0.12  # Match overall CLIP threshold (user-configured)

                            # Track if variance check failed (prevent validation from overwriting)
                            variance_check_failed = False

                            if view_variance > VARIANCE_THRESHOLD:
                                logger.warning(f"  ⚠️  INCONSISTENT GEOMETRY DETECTED")
                                logger.warning(f"     Variance {view_variance:.3f} > {VARIANCE_THRESHOLD} threshold")
                                logger.warning(f"     Model quality varies {view_variance*100:.0f}% across camera angles")
                                logger.warning(f"     Validators see random angles → will give Score=0.0")
                                passes = False
                                score = min_view_score  # Report worst view as the score
                                variance_check_failed = True

                            if min_view_score < MIN_ACCEPTABLE_VIEW:
                                logger.warning(f"  ⚠️  WORST VIEW TOO LOW")
                                logger.warning(f"     Min view CLIP {min_view_score:.3f} < {MIN_ACCEPTABLE_VIEW}")
                                logger.warning(f"     At least one angle looks terrible")
                                passes = False
                                score = min_view_score
                                variance_check_failed = True
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

                # Only run final validation if variance check passed
                # (prevents overwriting passes=False from variance check)
                if 'variance_check_failed' in locals() and variance_check_failed:
                    logger.info(f"  ⚠️  Skipping final validation (variance check failed)")
                    logger.info(f"  📊 Using worst-view score: {score:.3f}")
                else:
                    passes, score = app.state.clip_validator.validate_image(
                        validation_image,
                        validation_prompt
                    )
                app.state.clip_validator.to_cpu()
    
                t6 = time.time()
                logger.info(f"  ✅ Validation done ({t6-t5:.2f}s)")
                logger.info(f"  📊 DIAGNOSTIC - 3D Render CLIP score: {score:.3f}")
                logger.info(f"  📊 DIAGNOSTIC - Quality loss: {((flux_score - score) / flux_score * 100):.1f}%")

                # Log validation timing and CLIP score
                if log_id and hasattr(app.state, 'data_logger') and app.state.data_logger:
                    try:
                        app.state.data_logger.log_timing(log_id, "validation_time", t6-t5)
                        # Update output with CLIP score
                        app.state.data_logger.log_output(
                            log_id,
                            clip_score=score,
                            clip_threshold_pass=passes,
                            validation_pass=passes,
                        )
                    except Exception as e:
                        logger.debug(f"Data logger error (non-fatal): {e}")

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

                # Log rejection
                if log_id and hasattr(app.state, 'data_logger') and app.state.data_logger:
                    try:
                        app.state.data_logger.log_submission(
                            log_id,
                            submitted=False,
                            rejection_reason=f"clip_score_too_low_{score:.3f}"
                        )
                        app.state.data_logger.log_timing(log_id, "total_time", time.time() - t_start)
                        app.state.data_logger.finalize_generation(log_id)
                    except Exception as e:
                        logger.debug(f"Data logger error (non-fatal): {e}")

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
    
            # DIAGNOSTIC: Analyze PLY quality beyond just count (optional diagnostics)
            try:
                from diagnostics.ply_analyzer import analyze_gaussian_quality, diagnose_ply_issues

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

                # OLD TRACKER REMOVED: Now using GenerationDataLogger for comprehensive tracking

            except Exception as e:
                logger.error(f"  Diagnostic analysis failed: {e}")
    
            # Success!
            t_total = time.time() - t_start

            # Log submission and finalize
            if log_id and hasattr(app.state, 'data_logger') and app.state.data_logger:
                try:
                    # Determine submission status
                    submitted = True
                    if app.state.clip_validator:
                        submitted = passes

                    # Log submission
                    app.state.data_logger.log_submission(
                        log_id,
                        submitted=submitted,
                        rejection_reason=None if submitted else "clip_validation_failed"
                    )

                    # Log total time
                    app.state.data_logger.log_timing(log_id, "total_time", t_total)

                    # Finalize generation log (writes to disk)
                    app.state.data_logger.finalize_generation(log_id)
                    logger.debug(f"📊 Finalized generation log: {log_id}")
                except Exception as e:
                    logger.error(f"Error finalizing log (non-fatal): {e}")

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
            logger.error(f"❌ Generation failed: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")

            # Log failure
            if log_id and hasattr(app.state, 'data_logger') and app.state.data_logger:
                try:
                    error_type = type(e).__name__
                    app.state.data_logger.log_failure(
                        log_id,
                        error_type=error_type,
                        error_message=str(e),
                        stack_trace=traceback.format_exc()
                    )
                    app.state.data_logger.log_submission(
                        log_id,
                        submitted=False,
                        rejection_reason=f"generation_error_{error_type}"
                    )
                    app.state.data_logger.log_timing(log_id, "total_time", time.time() - t_start)
                    app.state.data_logger.finalize_generation(log_id)
                except Exception as log_err:
                    logger.error(f"Error logging failure (non-fatal): {log_err}")

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
