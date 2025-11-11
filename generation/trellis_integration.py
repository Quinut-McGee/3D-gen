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
import numpy as np
from loguru import logger
from PIL import Image, ImageFilter, ImageEnhance

# Import PLY fixes: bbox normalization and opacity corruption
from diagnostics.ply_fixer import fix_opacity_corruption, normalize_bounding_box


def enhance_image_for_trellis(rgba_image):
    """
    Enhance image detail to ensure TRELLIS generates dense voxel grids.

    More visual detail → more surface features detected → more active voxels → more gaussians

    This prevents sparse generations (57K gaussians) by giving TRELLIS more surface features
    to detect, even for geometrically simple objects.

    Args:
        rgba_image: PIL Image (RGBA)

    Returns:
        Enhanced PIL Image (RGBA) with more visual detail
    """
    logger.debug("  Enhancing image detail for dense voxel generation...")

    # 1. Enhance fine details (brings out texture)
    enhanced = rgba_image.filter(ImageFilter.DETAIL)

    # 2. Sharpen edges - SUBTLE enhancement to avoid artifacts
    sharpener = ImageEnhance.Sharpness(enhanced)
    enhanced = sharpener.enhance(1.5)  # 1.5x - subtle/natural level (reduced from 3.5x)

    # 3. Increase contrast - SUBTLE enhancement for surface detail
    contrast = ImageEnhance.Contrast(enhanced)
    enhanced = contrast.enhance(1.2)  # 1.2x - subtle level (reduced from 1.8x)

    logger.debug("  ✅ Image enhanced: sharpness 1.5x, contrast 1.2x (subtle values)")
    return enhanced


def apply_retry_enhancement(rgba_image):
    """
    Apply MODERATE alternative enhancement for retry attempts.

    IMPORTANT: Testing showed that EXTREME enhancement (5x sharpness, 2.5x contrast)
    actually makes things WORSE by creating artifacts. This uses moderate values
    that are DIFFERENT from the first attempt but not overly aggressive.

    Strategy: Focus on edge detection rather than over-sharpening.
    """
    logger.debug("  🔄 Applying alternative enhancement for retry...")

    # 1. Start with edge enhancement (different from first attempt)
    enhanced = rgba_image.filter(ImageFilter.EDGE_ENHANCE)

    # 2. Moderate sharpness (4.0x - between first attempt 3.5x and extreme)
    sharpener = ImageEnhance.Sharpness(enhanced)
    enhanced = sharpener.enhance(4.0)  # 4.0x - moderate increase from 3.5x

    # 3. Slightly higher contrast than first attempt
    contrast = ImageEnhance.Contrast(enhanced)
    enhanced = contrast.enhance(2.0)  # 2.0x - moderate contrast

    # 4. Slight brightness boost (helps with edge detection)
    brightness = ImageEnhance.Brightness(enhanced)
    enhanced = brightness.enhance(1.15)  # 1.15x - subtle brightness boost

    # 5. Final detail filter
    enhanced = enhanced.filter(ImageFilter.DETAIL)

    logger.debug("  ✅ Retry enhancement applied: sharpness 4.0x, contrast 2.0x, edge enhance, brightness 1.15x")
    return enhanced


def apply_depth_conditioning(rgba_image: Image.Image, depth_map: np.ndarray) -> Image.Image:
    """
    Apply depth-aware preprocessing to guide TRELLIS reconstruction.

    Strategy: Use depth map to enhance edges at depth boundaries,
    making it easier for TRELLIS to detect 3D structure.

    This solves three failure modes:
    1. Flat/thin objects (pruning shears with y=0.14)
    2. Ambiguous IMAGE-TO-3D tasks (z=0.09)
    3. Complex multi-object scenes (fruit salad occlusions)

    Args:
        rgba_image: PIL Image (RGBA)
        depth_map: Numpy array (H, W) with depth [0, 1]

    Returns:
        Enhanced PIL Image (RGBA) with depth cues
    """
    from scipy.ndimage import sobel

    logger.debug("  🎯 Applying depth-conditioned preprocessing...")

    # 1. Compute depth edges (where depth changes rapidly)
    depth_edges_x = sobel(depth_map, axis=1)
    depth_edges_y = sobel(depth_map, axis=0)
    depth_edges = np.hypot(depth_edges_x, depth_edges_y)

    # Normalize edges to [0, 1]
    if depth_edges.max() > 0:
        depth_edges = depth_edges / depth_edges.max()

    # 2. Enhance RGB edges at depth boundaries
    # Convert RGBA to numpy
    rgba_array = np.array(rgba_image).astype(np.float32) / 255.0

    # Apply subtle edge enhancement (1.5x at depth boundaries)
    enhancement_map = 1.0 + 0.5 * depth_edges
    for c in range(3):  # RGB channels
        rgba_array[:, :, c] = np.clip(
            rgba_array[:, :, c] * enhancement_map,
            0.0, 1.0
        )

    # 3. Modulate alpha channel by depth (optional)
    # Objects closer (depth=0) get slightly higher alpha
    # This helps TRELLIS prioritize foreground geometry
    depth_alpha_boost = 1.0 + 0.2 * (1.0 - depth_map)
    rgba_array[:, :, 3] = np.clip(
        rgba_array[:, :, 3] * depth_alpha_boost,
        0.0, 1.0
    )

    # Convert back to PIL
    rgba_enhanced = (rgba_array * 255).astype(np.uint8)
    enhanced_image = Image.fromarray(rgba_enhanced, mode='RGBA')

    logger.debug("  ✅ Depth conditioning applied (edge enhancement + alpha modulation)")
    return enhanced_image


def normalize_gaussian_scales(gs_model, target_scale_range=(0.01, 0.04)):
    """
    Normalize gaussian scales to proper range while preserving relative sizes.

    CRITICAL FIX: TRELLIS saves scales that are ~1000x too small (0.002-0.004)
    due to GaussianModel.load_ply() applying unnecessary torch.log()
    This function detects and corrects the corruption before normalization.

    Args:
        gs_model: GaussianModel with loaded PLY data
        target_scale_range: (min, max) tuple for target scale range

    Returns:
        gs_model with normalized scales
    """
    import torch

    # Get current scales BEFORE normalization
    current_scales = gs_model.get_scaling.detach()

    # CRITICAL FIX: Check if scales are suspiciously small (TRELLIS bug symptom)
    scale_magnitudes = torch.norm(current_scales, dim=1)
    current_avg = scale_magnitudes.mean().item()

    logger.info(f"  📏 Scale normalization:")
    logger.info(f"     Raw average scale: {current_avg:.6f}")

    # If avg scale is < 0.01, it's likely corrupted by double-log bug
    # TRELLIS native scales should be ~10.0, not ~0.003
    if current_avg < 0.01:
        logger.warning(f"     ⚠️  Suspiciously small scales detected ({current_avg:.6f})")
        logger.warning(f"     Applying 1000x correction (likely double-log bug)")

        # Undo the excessive compression by multiplying up
        # This is a heuristic - adjust the multiplier if needed
        correction_factor = target_scale_range[0] / current_avg  # Scale up to target minimum
        current_scales = current_scales * correction_factor
        scale_magnitudes = torch.norm(current_scales, dim=1)
        current_avg = scale_magnitudes.mean().item()

        logger.info(f"     After correction: {current_avg:.6f}")

    # Now proceed with normal normalization
    current_min = scale_magnitudes.min().item()
    current_max = scale_magnitudes.max().item()

    logger.info(f"     Before normalization: avg={current_avg:.3f}, range=[{current_min:.3f}, {current_max:.3f}]")

    # Calculate scaling factor
    target_avg = (target_scale_range[0] + target_scale_range[1]) / 2
    scale_factor = target_avg / current_avg

    # Apply uniform scaling (preserves relative sizes)
    normalized_scales = current_scales * scale_factor

    # Clamp to target range to prevent outliers
    normalized_scales = torch.clamp(
        normalized_scales,
        min=target_scale_range[0],
        max=target_scale_range[1]
    )

    # Update model (convert back to log space - GaussianModel stores scales in log space)
    gs_model._scaling = torch.log(normalized_scales)

    # Verify results
    new_scale_magnitudes = torch.norm(normalized_scales, dim=1)
    new_avg = new_scale_magnitudes.mean().item()

    logger.info(f"     After normalization: avg={new_avg:.3f}, range=[{target_scale_range[0]:.3f}, {target_scale_range[1]:.3f}]")
    logger.info(f"     Scale factor: {scale_factor:.4f} ({current_avg:.3f} → {new_avg:.3f})")

    return gs_model


async def _call_trellis_api(rgb_image, trellis_url, timeout=60.0):
    """
    Helper function to call TRELLIS API.

    Args:
        rgb_image: PIL Image (RGB mode)
        trellis_url: TRELLIS microservice URL
        timeout: Request timeout in seconds

    Returns:
        result dict from TRELLIS API
    """
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
    async with httpx.AsyncClient(timeout=timeout) as client:
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

    return result


async def generate_with_trellis(rgba_image, prompt, trellis_url="http://localhost:10008", enable_scale_normalization=False, enable_image_enhancement=False, min_gaussians=0, depth_map=None):
    """
    Generate 3D Gaussian Splat using TRELLIS microservice.

    Includes retry logic: If first attempt produces sparse output (<150K gaussians),
    retry with extreme image enhancement to force denser voxel generation.

    Args:
        rgba_image: PIL Image (RGBA) from FLUX → background removal
        prompt: Text prompt for logging/debugging (not used by TRELLIS)
        trellis_url: TRELLIS microservice URL
        enable_scale_normalization: Enable scale normalization correction (diagnostic mode: default OFF)
        enable_image_enhancement: Enable image enhancement before TRELLIS (diagnostic mode: default OFF)
        min_gaussians: Minimum gaussian count threshold (0 = disabled, 150000 = strict quality gate)
        depth_map: Optional numpy array (H, W) with depth values [0, 1]
                  If provided, will be used to guide 3D reconstruction

    Returns:
        ply_bytes: Binary PLY data
        gs_model: GaussianModel for validation (loaded from PLY)
        timings: Dict of timing info
    """

    # Step 3: Direct Gaussian generation with TRELLIS microservice (5s)
    t3_start = time.time()
    logger.info("  [3/4] Generating 3D Gaussians with TRELLIS microservice...")

    # Quality threshold (0 = disabled in diagnostic mode)
    MIN_GAUSSIANS = min_gaussians  # Configurable threshold (default 0 = no gate)

    # Keep original image for potential retry
    original_rgba = rgba_image.copy()

    try:
        # DEPTH CONDITIONING: Apply depth-aware preprocessing if depth map provided
        if depth_map is not None:
            rgba_image = apply_depth_conditioning(rgba_image, depth_map)
            logger.info("  🎯 Depth conditioning applied (edge enhancement at depth boundaries)")

        # ATTEMPT 1: Image enhancement (if enabled)
        # DIAGNOSTIC MODE: Enhancement is OPTIONAL (default OFF to match template)
        # When enabled: 3.5x sharpness, 1.8x contrast for denser voxel detection
        if enable_image_enhancement:
            enhanced_rgba = enhance_image_for_trellis(rgba_image)
            logger.debug("  Image enhancement applied (3.5x sharpness, 1.8x contrast)")
        else:
            enhanced_rgba = rgba_image
            logger.debug("  Image enhancement DISABLED (diagnostic mode - using raw RGBA)")

        # Convert RGBA to RGB (TRELLIS expects RGB)
        if enhanced_rgba.mode == 'RGBA':
            rgb_image = enhanced_rgba.convert('RGB')
        else:
            rgb_image = enhanced_rgba

        # ATTEMPT 1: Call TRELLIS with standard enhancement
        logger.debug(f"  Calling TRELLIS (attempt 1/2) at {trellis_url}/generate...")
        result = await _call_trellis_api(rgb_image, trellis_url, timeout=60.0)

        # Decode PLY from base64
        ply_bytes = base64.b64decode(result["ply_base64"])
        num_gaussians = result['num_gaussians']

        t3_end = time.time()
        logger.info(f"  ✅ TRELLIS generation done ({t3_end-t3_start:.2f}s, service: {result['generation_time']:.2f}s)")
        logger.info(f"     Gaussians: {num_gaussians:,}, File size: {result['file_size_mb']:.1f} MB")

        # QUALITY GATE WITH RETRY: Check if output is too sparse (only if gate is enabled)
        if MIN_GAUSSIANS > 0 and num_gaussians < MIN_GAUSSIANS:
            logger.warning(f"⚠️  SPARSE GENERATION: {num_gaussians:,} < {MIN_GAUSSIANS:,} threshold")
            logger.warning(f"   Retrying with alternative enhancement (attempt 2/2)...")

            # ATTEMPT 2: Retry with moderate alternative enhancement + different seed
            try:
                retry_rgba = apply_retry_enhancement(original_rgba)

                # Convert to RGB
                if retry_rgba.mode == 'RGBA':
                    retry_rgb = retry_rgba.convert('RGB')
                else:
                    retry_rgb = retry_rgba

                # Retry TRELLIS call
                retry_result = await _call_trellis_api(retry_rgb, trellis_url, timeout=60.0)

                retry_gaussians = retry_result['num_gaussians']
                logger.info(f"  🔄 Retry result: {retry_gaussians:,} gaussians (was {num_gaussians:,})")

                # Check if retry improved the result
                if retry_gaussians >= MIN_GAUSSIANS:
                    # Retry succeeded! Use the retry result
                    logger.info(f"  ✅ RETRY SUCCESSFUL: {retry_gaussians:,} >= {MIN_GAUSSIANS:,}")
                    ply_bytes = base64.b64decode(retry_result["ply_base64"])
                    result = retry_result  # Use retry result for stats
                    num_gaussians = retry_gaussians
                elif retry_gaussians > num_gaussians:
                    # Retry improved but still not enough - use better result
                    logger.warning(f"  ⚠️  RETRY IMPROVED but still sparse: {retry_gaussians:,} < {MIN_GAUSSIANS:,}")
                    logger.warning(f"     Using improved result anyway (+{retry_gaussians - num_gaussians:,} gaussians)")
                    ply_bytes = base64.b64decode(retry_result["ply_base64"])
                    result = retry_result
                    num_gaussians = retry_gaussians
                else:
                    # Retry didn't help - reject
                    logger.error(f"  ❌ RETRY FAILED: {retry_gaussians:,} gaussians (no improvement)")
                    logger.error(f"     This would receive Score=0.0 from validators - rejecting to avoid penalty")
                    raise ValueError(f"Insufficient gaussian density after retry: {retry_gaussians:,} gaussians (minimum: {MIN_GAUSSIANS:,})")

            except Exception as retry_error:
                logger.error(f"  ❌ Retry attempt failed: {retry_error}")
                logger.error(f"     Original sparse generation: {num_gaussians:,} < {MIN_GAUSSIANS:,}")
                raise ValueError(f"Insufficient gaussian density: {num_gaussians:,} gaussians (minimum: {MIN_GAUSSIANS:,}, retry failed)")

        # Final quality check (only if gate is enabled)
        if MIN_GAUSSIANS > 0 and num_gaussians < MIN_GAUSSIANS:
            logger.error(f"❌ QUALITY GATE FAILED: {num_gaussians:,} < {MIN_GAUSSIANS:,} threshold!")
            logger.error(f"   This would receive Score=0.0 from validators - rejecting to avoid penalty")
            logger.error(f"   Prompt: '{prompt[:100]}...'")
            raise ValueError(f"Insufficient gaussian density: {num_gaussians:,} gaussians (minimum: {MIN_GAUSSIANS:,})")

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

        # 🔬 DIAGNOSTIC: Inspect raw TRELLIS PLY format BEFORE GaussianModel.load_ply() processes it
        try:
            from plyfile import PlyData
            plydata = PlyData.read(tmp_path)
            raw_scales = plydata['vertex']['scale_0'][:10]  # First 10 gaussians
            logger.info(f"  🔬 Raw TRELLIS PLY scales (first 10): {raw_scales}")
            logger.info(f"  🔬 Min/Max/Avg: {float(raw_scales.min()):.6f} / {float(raw_scales.max()):.6f} / {float(raw_scales.mean()):.6f}")
        except Exception as e:
            logger.warning(f"  Could not inspect raw PLY: {e}")

        # Load into GaussianModel
        gs_model = GaussianModel(sh_degree=2)  # TRELLIS uses SH degree 2
        gs_model.load_ply(tmp_path)

        # 🔬 DIAGNOSTIC: Check scales AFTER GaussianModel.load_ply() processes them
        try:
            loaded_internal_scales = gs_model._scaling[:10]  # Internal log-space representation
            loaded_exp_scales = gs_model.get_scaling[:10]    # After exp() activation
            logger.info(f"  🔬 After load_ply() - Internal (_scaling): {loaded_internal_scales.cpu().numpy()}")
            logger.info(f"  🔬 After load_ply() - Exp-space (get_scaling): {loaded_exp_scales.cpu().numpy()}")
            logger.info(f"  🔬 Avg internal: {gs_model._scaling.mean().item():.6f}, Avg exp-space: {gs_model.get_scaling.mean().item():.6f}")
        except Exception as e:
            logger.warning(f"  Could not inspect loaded scales: {e}")

        # CRITICAL FIX #1: Normalize oversized bounding boxes (30-40% of rejections)
        # Validators have ZERO tolerance for bbox dimensions > 1.0
        # This fix scales models to fit within 0.98 unit cube while preserving aspect ratio
        logger.info("  🔧 Checking bounding box dimensions...")
        gs_model = normalize_bounding_box(gs_model)

        # CRITICAL FIX #2: Fix opacity corruption (inf/NaN values from TRELLIS)
        # 50% of TRELLIS generations have corrupted opacities → Score=0.0
        # This fix corrects them before submission
        logger.info("  🔧 Checking for opacity corruption...")
        gs_model = fix_opacity_corruption(gs_model)

        # Save the fixed model back to PLY and update ply_bytes
        # This ensures we submit the FIXED PLY, not the corrupted original
        fixed_tmp_path = tmp_path.replace('.ply', '_opacity_fixed.ply')
        gs_model.save_ply(fixed_tmp_path)

        # Re-read the fixed PLY to update ply_bytes
        with open(fixed_tmp_path, 'rb') as f:
            ply_bytes = f.read()  # Update ply_bytes with fixed PLY

        logger.info(f"  ✅ Fixed PLY ready for submission ({len(ply_bytes) / (1024*1024):.1f} MB)")

        # Clean up original temp file, keep using fixed version
        os.unlink(tmp_path)
        tmp_path = fixed_tmp_path  # Update path for subsequent operations

        # DIAGNOSTIC MODE: Scale normalization is OPTIONAL
        # By default (enable_scale_normalization=False), we submit TRELLIS PLY as-is (like official template)
        # This lets us test if our "fixes" are actually helping or hurting
        if enable_scale_normalization:
            # CRITICAL FIX: Normalize gaussian scales to validator-friendly range
            # TRELLIS generates avg_scale ~11.5 (arbitrary units)
            # Validators expect avg_scale 0.005-0.05
            # This fixes Score=0.0 rejections despite passing CLIP validation
            gs_model = normalize_gaussian_scales(gs_model, target_scale_range=(0.01, 0.04))

            # CRITICAL: Save normalized model back to PLY and update ply_bytes
            # Without this, we'd send the ORIGINAL (unnormalized) PLY to validators!
            normalized_tmp_path = tmp_path.replace('.ply', '_normalized.ply')
            gs_model.save_ply(normalized_tmp_path)

            # Re-read the normalized PLY to update ply_bytes
            with open(normalized_tmp_path, 'rb') as f:
                ply_bytes = f.read()  # Update the ply_bytes variable with normalized PLY

            logger.info(f"  ✅ Normalized PLY saved ({len(ply_bytes) / (1024*1024):.1f} MB)")

            # Clean up temp files
            os.unlink(tmp_path)
            os.unlink(normalized_tmp_path)
        else:
            logger.info(f"  ⚠️  Scale normalization DISABLED - submitting raw TRELLIS output (diagnostic mode)")
            # Clean up temp file
            os.unlink(tmp_path)

        t4_end = time.time()
        logger.info(f"  ✅ Gaussian model loaded ({t4_end-t4_start:.2f}s)")

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        logger.error(f"Failed to load GaussianModel for validation: {e}")
        logger.error(f"Full traceback:\n{traceback_str}")
        logger.warning("Continuing without validation model (PLY is still valid)")
        gs_model = None
        t4_end = time.time()

    timings = {
        "trellis": t3_end - t3_start,
        "model_load": t4_end - t4_start if gs_model else 0.0,
        "total_3d": t4_end - t3_start,
        "num_gaussians": result['num_gaussians'],
        "file_size_mb": result['file_size_mb']
    }

    return ply_bytes, gs_model, timings
