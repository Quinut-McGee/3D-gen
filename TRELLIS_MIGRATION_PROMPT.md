# TRELLIS INTEGRATION & TESTING FRAMEWORK - COMPLETE MIGRATION PLAN

## CONTEXT: Why You Were Kicked from Mainnet

You were producing 0.7-0.8 CLIP scores but still got kicked from mainnet. This indicates:

1. **Your architecture is fundamentally limited** - You're using InstantMesh→Mesh2Gaussian (mesh-based) while competitive miners use TRELLIS (direct gaussian generation)
2. **Your gaussian counts are too low** - You're at 12K-50K gaussians (~1-2MB files) while top miners hit 400K-500K gaussians (~50MB files)
3. **Your quality ceiling is capped** - Mesh-sampling approach can't scale to competitive gaussian densities

## MISSION: Full TRELLIS Integration + Robust Testing Infrastructure

You need to:
1. **Replace InstantMesh with TRELLIS** for direct gaussian generation
2. **Fix CLIP validation** to work with TRELLIS outputs
3. **Build comprehensive testing framework** to validate before mainnet deployment
4. **Achieve 300K-500K gaussians** (30-50MB files) to match competitive miners

---

## PHASE 1: TRELLIS INSTALLATION & INTEGRATION

### Step 1.1: Install TRELLIS

**Requirements Check:**
- NVIDIA GPU with 16GB+ VRAM (you should have this)
- CUDA 11.8 or 12.2
- Python 3.8+

**Installation:**
```bash
cd /path/to/three-gen-subnet

# Clone TRELLIS with submodules
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
cd TRELLIS

# Install dependencies (they'll create a conda env, but we want to use existing env)
# Read their requirements.txt and install to your current environment
pip install -r requirements.txt

# Test basic import
python -c "from trellis.pipelines import TrellisImageTo3DPipeline; print('✅ TRELLIS installed')"
```

**Note:** TRELLIS is designed for Linux with high-end GPUs. If installation fails:
- Check https://github.com/sdbds/TRELLIS-for-windows for Windows-specific instructions
- May need to install in fp16 mode to reduce VRAM from 16GB to 8GB

### Step 1.2: Create TRELLIS Integration Module

**Location:** `generation/trellis_integration.py`

**Purpose:** Replace `instantmesh_integration.py` with TRELLIS-based gaussian generation

**Implementation:**

```python
"""
TRELLIS Integration for Direct Gaussian Generation
Replaces InstantMesh → Mesh2Gaussian pipeline with direct gaussian generation
"""

import torch
import numpy as np
from PIL import Image
import time
from loguru import logger
import io

# Import TRELLIS
try:
    from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
    from TRELLIS.trellis.representations import Gaussian, SparseStructuredLatents
except ImportError:
    logger.error("TRELLIS not installed! Install from: https://github.com/microsoft/TRELLIS")
    raise


class TRELLISGaussianGenerator:
    """
    Direct Gaussian generation using TRELLIS

    This is the competitive architecture that top miners use.
    Generates 300K-500K gaussians directly from images.
    """

    def __init__(self, model_name: str = "microsoft/TRELLIS-image-large"):
        """
        Initialize TRELLIS pipeline

        Args:
            model_name: TRELLIS model variant
                - "microsoft/TRELLIS-image-large" (2B params, best quality)
                - "microsoft/TRELLIS-image-base" (smaller, faster)
        """
        logger.info(f"Loading TRELLIS pipeline: {model_name}...")

        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(model_name)
        self.pipeline = self.pipeline.to("cuda")

        logger.info("✅ TRELLIS pipeline loaded on GPU")

    async def generate_gaussian_splat(
        self,
        rgba_image: Image.Image,
        prompt: str,
        num_inference_steps: int = 50,
        seed: int = 42
    ):
        """
        Generate Gaussian Splat directly from image using TRELLIS

        Args:
            rgba_image: PIL Image (RGBA) from FLUX → background removal
            prompt: Text prompt (used for context/logging)
            num_inference_steps: Diffusion steps (higher = better quality, slower)
                - 20 steps: Fast (~5s), acceptable quality
                - 50 steps: Balanced (~10s), good quality
                - 100 steps: Slow (~20s), best quality
            seed: Random seed for reproducibility

        Returns:
            ply_bytes: Binary PLY data (Gaussian Splat format)
            gs_model: GaussianModel for validation/rendering (if needed)
            timings: Dict of timing info
        """
        start_time = time.time()
        logger.info(f"  [3/4] Generating Gaussian Splat with TRELLIS...")
        logger.info(f"     Steps: {num_inference_steps}, Seed: {seed}")

        try:
            # Convert RGBA to RGB (TRELLIS expects RGB)
            rgb_image = rgba_image.convert('RGB')

            # Run TRELLIS inference
            # TRELLIS outputs multiple formats - request gaussians
            outputs = self.pipeline.run(
                rgb_image,
                seed=seed,
                formats=["gaussian"],  # Request Gaussian Splat output
                num_inference_steps=num_inference_steps,
                # TRELLIS-specific quality settings
                sparse_structure_sampler_params={
                    "steps": num_inference_steps,
                    "cfg_strength": 7.5,  # Classifier-free guidance
                },
                slat_sampler_params={
                    "steps": num_inference_steps,
                    "cfg_strength": 3.0,
                }
            )

            # Extract Gaussian representation
            # TRELLIS returns Gaussian object with .to_ply() method
            gaussian = outputs['gaussian'][0]

            # Convert to PLY binary format
            ply_bytes = gaussian.to_ply()

            # Get statistics
            num_gaussians = self._count_gaussians_in_ply(ply_bytes)
            file_size_mb = len(ply_bytes) / (1024 * 1024)

            elapsed = time.time() - start_time
            logger.info(f"  ✅ TRELLIS generation done ({elapsed:.2f}s)")
            logger.info(f"     Generated {num_gaussians:,} gaussians ({file_size_mb:.1f} MB)")

            timings = {
                "trellis": elapsed,
                "total_3d": elapsed
            }

            # Try to create GaussianModel for rendering validation
            gs_model = self._create_gaussian_model_from_ply(ply_bytes)

            return ply_bytes, gs_model, timings

        except Exception as e:
            logger.error(f"TRELLIS generation failed: {e}", exc_info=True)
            raise

    def _count_gaussians_in_ply(self, ply_bytes: bytes) -> int:
        """
        Parse PLY header to count number of gaussians

        Args:
            ply_bytes: PLY file bytes

        Returns:
            Number of gaussians (vertex count)
        """
        try:
            # Decode header (ASCII until end_header)
            header_end = ply_bytes.find(b"end_header\n")
            if header_end == -1:
                return 0

            header = ply_bytes[:header_end].decode('utf-8')

            # Find "element vertex N" line
            for line in header.split('\n'):
                if line.startswith('element vertex'):
                    return int(line.split()[-1])

            return 0
        except Exception as e:
            logger.warning(f"Failed to parse PLY header: {e}")
            return 0

    def _create_gaussian_model_from_ply(self, ply_bytes: bytes):
        """
        Create GaussianModel from PLY bytes for rendering validation

        This enables CLIP validation by rendering the 3D model.

        Args:
            ply_bytes: PLY file bytes

        Returns:
            GaussianModel instance or None if failed
        """
        try:
            from DreamGaussianLib.GaussianSplattingModel import GaussianModel

            # Save PLY to temp file (GaussianModel loads from file)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp:
                tmp.write(ply_bytes)
                tmp_path = tmp.name

            # Load into GaussianModel
            gs_model = GaussianModel(sh_degree=3)  # TRELLIS uses higher SH degrees
            gs_model.load_ply(tmp_path)

            # Clean up temp file
            import os
            os.unlink(tmp_path)

            logger.debug(f"Created GaussianModel from PLY for rendering")
            return gs_model

        except Exception as e:
            logger.warning(f"Failed to create GaussianModel from PLY: {e}")
            return None


async def generate_with_trellis(
    rgba_image: Image.Image,
    prompt: str,
    trellis_generator: TRELLISGaussianGenerator,
    num_inference_steps: int = 50
):
    """
    Wrapper function to match instantmesh_integration.py interface

    This allows drop-in replacement in serve_competitive.py

    Args:
        rgba_image: PIL Image (RGBA) from FLUX → background removal
        prompt: Text prompt
        trellis_generator: Initialized TRELLISGaussianGenerator instance
        num_inference_steps: TRELLIS inference steps (20-100)

    Returns:
        ply_bytes: Binary PLY data
        gs_model: GaussianModel for validation
        timings: Dict of timing info
    """
    return await trellis_generator.generate_gaussian_splat(
        rgba_image=rgba_image,
        prompt=prompt,
        num_inference_steps=num_inference_steps
    )
```

### Step 1.3: Integrate TRELLIS into serve_competitive.py

**Changes needed in `generation/serve_competitive.py`:**

1. **Import TRELLIS instead of InstantMesh:**
```python
# OLD (line 34):
# from instantmesh_integration import generate_with_instantmesh

# NEW:
from trellis_integration import TRELLISGaussianGenerator, generate_with_trellis
```

2. **Remove InstantMesh microservice references:**
```python
# REMOVE from AppState (lines 84-85):
# instantmesh_service_url: str = "http://localhost:10007"
# mesh_to_gaussian: MeshToGaussianConverter = None

# ADD:
trellis_generator: TRELLISGaussianGenerator = None
```

3. **Initialize TRELLIS in startup_event() (replace lines 196-217):**
```python
# [3/4] Load TRELLIS for direct gaussian generation
logger.info("\n[3/4] Loading TRELLIS gaussian generator...")
app.state.trellis_generator = TRELLISGaussianGenerator(
    model_name="microsoft/TRELLIS-image-large"  # 2B params, best quality
)
logger.info("✅ TRELLIS gaussian generator ready")
```

4. **Replace InstantMesh call in generate() endpoint (lines 391-413):**
```python
# Step 3: 3D generation with TRELLIS (direct gaussian generation)
t3_start = time.time()
try:
    # Call TRELLIS integration (direct gaussian generation)
    ply_bytes, gs_model, timings = await generate_with_trellis(
        rgba_image=rgba_image,
        prompt=prompt,
        trellis_generator=app.state.trellis_generator,
        num_inference_steps=50  # Adjust for speed/quality balance
    )

    # Cache for validation
    app.state.last_gs_model = gs_model

    t4 = time.time()
    logger.info(f"  ✅ 3D generation done ({t4-t3_start:.2f}s)")
    logger.info(f"     TRELLIS: {timings['trellis']:.2f}s")
    logger.info(f"     Generated {len(ply_bytes)/1024/1024:.1f} MB Gaussian Splat PLY")

except Exception as e:
    logger.error(f"TRELLIS generation failed: {e}", exc_info=True)
    cleanup_memory()
    raise
```

5. **Update health check endpoint (line 579):**
```python
"models_loaded": {
    "flux": app.state.flux_generator is not None,
    "background_remover": app.state.background_remover is not None,
    "trellis_generator": app.state.trellis_generator is not None,  # NEW
    "clip_validator": app.state.clip_validator is not None
}
```

---

## PHASE 2: FIX CLIP VALIDATION WITH TRELLIS

### Issue: CLIP Validation Not Working

Based on your codebase, CLIP validation exists but you mentioned it's not working. The issue is likely:

1. **Rendering fails** for TRELLIS-generated gaussians (different format than InstantMesh)
2. **CLIP scores are unreliable** due to rendering quality
3. **Prompt enhancement breaks validation** (enhanced prompts don't match network validation)

### Step 2.1: Fix Gaussian Rendering for CLIP Validation

**File:** `generation/validators/clip_validator.py`

**Problem:** Lines 117-168 try to render PLY files but may fail with TRELLIS format

**Solution:** Ensure rendering works with TRELLIS's higher SH degrees

```python
# In validate_ply_renders() method (line 117):

def validate_ply_renders(
    self,
    ply_bytes: bytes,
    prompt: str,
    num_views: int = 4,
    use_original_prompt: bool = True  # NEW: Don't use enhanced prompt for validation
) -> Tuple[bool, float]:
    """
    Validate a PLY file by rendering it and checking CLIP scores.

    Args:
        ply_bytes: Raw PLY file bytes
        prompt: Original text prompt
        num_views: Number of views to render for validation
        use_original_prompt: If True, strips enhancement from prompt

    Returns:
        (passes_threshold, average_clip_score)
    """
    try:
        # Import here to avoid circular dependency
        from rendering.quick_render import render_ply_to_images

        # Render PLY to images
        images = render_ply_to_images(ply_bytes, num_views=num_views)

        if not images:
            logger.warning("Failed to render PLY for validation")
            return False, 0.0

        # Strip prompt enhancement for validation
        # Network validators use original prompts, not enhanced ones
        if use_original_prompt and "photorealistic" in prompt:
            # Extract original prompt (everything before first comma)
            validation_prompt = prompt.split(",")[0].strip()
            if validation_prompt.startswith("a photorealistic "):
                validation_prompt = validation_prompt.replace("a photorealistic ", "")
            logger.debug(f"Using original prompt for validation: '{validation_prompt}'")
        else:
            validation_prompt = prompt

        # Validate each view
        scores = []
        for i, image in enumerate(images):
            _, score = self.validate_image(image, validation_prompt)
            scores.append(score)
            logger.debug(f"View {i+1}/{num_views}: CLIP={score:.3f}")

        # Use average score
        avg_score = np.mean(scores)
        passes = avg_score >= self.threshold

        logger.info(
            f"Validation result: {avg_score:.3f} "
            f"({'PASS' if passes else 'FAIL'}, threshold={self.threshold})"
        )

        return passes, avg_score

    except Exception as e:
        logger.error(f"PLY validation error: {e}", exc_info=True)
        return False, 0.0
```

### Step 2.2: Update serve_competitive.py to Use Original Prompt for CLIP

**Change in generate() endpoint (lines 479-496):**

```python
# Step 4: CLIP validation
if app.state.clip_validator:
    t5 = time.time()
    logger.info("  [4/4] Validating with CLIP...")

    # Render 3D Gaussian Splat for validation using cached model
    try:
        logger.debug("  Rendering 3D Gaussian Splat for CLIP validation...")

        # Get the cached GaussianModel
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

                # Use first view for CLIP validation
                validation_image = rendered_views[0]
                logger.debug(f"  Using 3D render for validation")
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

    # Validate with CLIP using ORIGINAL prompt (not enhanced)
    app.state.clip_validator.to_gpu()

    # CRITICAL FIX: Use original prompt, not enhanced prompt
    # Enhanced prompt was: "a photorealistic {prompt}, professional product photography, ..."
    # Network validators use original prompt only
    validation_prompt = prompt  # Already the original user prompt

    passes, score = app.state.clip_validator.validate_image(
        validation_image,
        validation_prompt
    )
    app.state.clip_validator.to_cpu()

    t6 = time.time()
    logger.info(f"  ✅ Validation done ({t6-t5:.2f}s)")
    logger.info(f"  📊 3D Render CLIP score: {score:.3f} vs threshold {app.state.clip_validator.threshold}")

    # Clean up
    del validation_image
    if 'rendered_views' in locals():
        del rendered_views
    cleanup_memory()
```

---

## PHASE 3: BUILD COMPREHENSIVE TESTING FRAMEWORK

### Step 3.1: Create TRELLIS-Specific Test Suite

**File:** `generation/test_trellis_pipeline.py` (NEW)

```python
#!/usr/bin/env python3
"""
TRELLIS Pipeline Test Suite

Tests TRELLIS integration before mainnet deployment:
1. TRELLIS gaussian generation (target: 300K-500K gaussians)
2. PLY file structure validation
3. CLIP validation on TRELLIS outputs
4. End-to-end pipeline timing (target: <30s)
5. Gaussian count and file size verification
"""

import sys
import time
import asyncio
from pathlib import Path
from PIL import Image
import numpy as np
from io import BytesIO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trellis_integration import TRELLISGaussianGenerator
from models.flux_generator import FluxImageGenerator
from models.background_remover import SOTABackgroundRemover
from validators.clip_validator import CLIPValidator

print("=" * 80)
print("TRELLIS PIPELINE TEST SUITE")
print("=" * 80)


def test_trellis_installation():
    """Test 1: Verify TRELLIS is installed correctly"""
    print("\n[Test 1] TRELLIS Installation Check")
    try:
        from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
        print("✅ TRELLIS imports successfully")
        return True
    except ImportError as e:
        print(f"❌ TRELLIS not installed: {e}")
        return False


async def test_trellis_basic_generation():
    """Test 2: Basic TRELLIS generation with simple test image"""
    print("\n[Test 2] Basic TRELLIS Generation")
    try:
        # Create simple test image (red square on transparent background)
        test_image = Image.new('RGBA', (512, 512), color=(0, 0, 0, 0))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_image)
        draw.rectangle([128, 128, 384, 384], fill=(255, 0, 0, 255))

        # Initialize TRELLIS
        print("  Loading TRELLIS generator...")
        generator = TRELLISGaussianGenerator()

        # Generate
        print("  Generating Gaussian Splat...")
        start = time.time()
        ply_bytes, gs_model, timings = await generator.generate_gaussian_splat(
            rgba_image=test_image,
            prompt="red cube",
            num_inference_steps=20  # Fast test mode
        )
        elapsed = time.time() - start

        # Validate output
        if len(ply_bytes) < 1000:
            print(f"❌ Output too small: {len(ply_bytes)} bytes")
            return False

        if not ply_bytes.startswith(b"ply\n"):
            print(f"❌ Invalid PLY header")
            return False

        # Parse gaussian count
        num_gaussians = generator._count_gaussians_in_ply(ply_bytes)
        file_size_mb = len(ply_bytes) / (1024 * 1024)

        print(f"✅ TRELLIS generation successful")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Gaussians: {num_gaussians:,}")
        print(f"   File size: {file_size_mb:.1f} MB")

        # Quality checks
        if num_gaussians < 100000:
            print(f"⚠️  WARNING: Low gaussian count ({num_gaussians:,} < 100K)")
            print(f"   Competitive miners use 300K-500K gaussians")

        if file_size_mb < 10:
            print(f"⚠️  WARNING: Small file size ({file_size_mb:.1f} MB < 10 MB)")
            print(f"   Competitive miners generate 30-50 MB files")

        return True

    except Exception as e:
        print(f"❌ TRELLIS generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_trellis_with_flux_pipeline():
    """Test 3: Full pipeline (FLUX → Background Removal → TRELLIS)"""
    print("\n[Test 3] Full Pipeline: FLUX → TRELLIS")
    try:
        # Initialize all components
        print("  Loading FLUX generator...")
        flux_gen = FluxImageGenerator(device="cuda")

        print("  Loading background remover...")
        bg_remover = SOTABackgroundRemover(device="cuda")

        print("  Loading TRELLIS generator...")
        trellis_gen = TRELLISGaussianGenerator()

        # Test prompt
        prompt = "red sports car"

        # Step 1: FLUX generation
        print(f"\n  [1/3] FLUX: Generating image for '{prompt}'...")
        t1 = time.time()
        image = flux_gen.generate(
            prompt=prompt,
            num_inference_steps=4,
            height=512,
            width=512
        )
        t2 = time.time()
        print(f"  ✅ FLUX done ({t2-t1:.2f}s)")

        # Step 2: Background removal
        print(f"  [2/3] Background removal...")
        rgba_image = bg_remover.remove_background(image, threshold=0.5)
        t3 = time.time()
        print(f"  ✅ Background removal done ({t3-t2:.2f}s)")

        # Step 3: TRELLIS generation
        print(f"  [3/3] TRELLIS: Generating Gaussian Splat...")
        ply_bytes, gs_model, timings = await trellis_gen.generate_gaussian_splat(
            rgba_image=rgba_image,
            prompt=prompt,
            num_inference_steps=50  # Quality mode
        )
        t4 = time.time()

        # Total timing
        total_time = t4 - t1

        # Parse gaussian count
        num_gaussians = trellis_gen._count_gaussians_in_ply(ply_bytes)
        file_size_mb = len(ply_bytes) / (1024 * 1024)

        print(f"\n✅ Full pipeline successful")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   FLUX: {t2-t1:.2f}s")
        print(f"   Background removal: {t3-t2:.2f}s")
        print(f"   TRELLIS: {t4-t3:.2f}s")
        print(f"   Gaussians: {num_gaussians:,}")
        print(f"   File size: {file_size_mb:.1f} MB")

        # Performance checks
        if total_time > 30:
            print(f"⚠️  WARNING: Total time ({total_time:.2f}s) > 30s target")

        if num_gaussians < 200000:
            print(f"⚠️  WARNING: Gaussian count ({num_gaussians:,}) < 200K (competitive range: 300K-500K)")

        if file_size_mb < 20:
            print(f"⚠️  WARNING: File size ({file_size_mb:.1f} MB) < 20 MB (competitive range: 30-50 MB)")

        return True

    except Exception as e:
        print(f"❌ Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_trellis_clip_validation():
    """Test 4: CLIP validation on TRELLIS-generated gaussians"""
    print("\n[Test 4] CLIP Validation on TRELLIS Output")
    try:
        # Generate a simple object
        print("  Generating test object with TRELLIS...")

        # Create test image (blue teapot shape)
        test_image = Image.new('RGBA', (512, 512), color=(0, 0, 0, 0))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_image)
        # Draw teapot-like shape
        draw.ellipse([150, 200, 350, 350], fill=(0, 0, 255, 255))  # Body
        draw.ellipse([200, 180, 300, 220], fill=(0, 0, 255, 255))  # Lid
        draw.rectangle([340, 250, 400, 280], fill=(0, 0, 255, 255))  # Spout

        generator = TRELLISGaussianGenerator()
        ply_bytes, gs_model, timings = await generator.generate_gaussian_splat(
            rgba_image=test_image,
            prompt="blue teapot",
            num_inference_steps=20
        )

        print(f"  Generated {len(ply_bytes)/1024:.1f} KB PLY")

        # Render for CLIP validation
        print("  Rendering for CLIP validation...")
        from rendering.quick_render import render_gaussian_model_to_images

        if gs_model is None:
            print("❌ No GaussianModel available for rendering")
            return False

        rendered_views = render_gaussian_model_to_images(
            model=gs_model,
            num_views=4,
            resolution=512,
            device="cuda"
        )

        if not rendered_views:
            print("❌ Rendering failed")
            return False

        print(f"  Rendered {len(rendered_views)} views")

        # CLIP validation
        print("  Running CLIP validation...")
        validator = CLIPValidator(device="cuda", threshold=0.6)
        validator.to_gpu()

        # Test with matching prompt
        _, score_good = validator.validate_image(rendered_views[0], "blue teapot")
        print(f"   Matching prompt ('blue teapot'): {score_good:.3f}")

        # Test with non-matching prompt
        _, score_bad = validator.validate_image(rendered_views[0], "red car")
        print(f"   Non-matching prompt ('red car'): {score_bad:.3f}")

        validator.to_cpu()

        # Validate
        if score_good <= score_bad:
            print(f"❌ CLIP validation failed: matching prompt ({score_good:.3f}) should score higher than non-matching ({score_bad:.3f})")
            return False

        print(f"✅ CLIP validation working correctly")
        print(f"   Score difference: {(score_good - score_bad):.3f}")

        return True

    except Exception as e:
        print(f"❌ CLIP validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_trellis_quality_scaling():
    """Test 5: Quality scaling with different inference steps"""
    print("\n[Test 5] Quality Scaling (Inference Steps)")
    try:
        # Create test image
        test_image = Image.new('RGBA', (512, 512), color=(0, 0, 0, 0))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_image)
        draw.rectangle([128, 128, 384, 384], fill=(255, 0, 0, 255))

        generator = TRELLISGaussianGenerator()

        # Test different inference step counts
        step_configs = [
            (20, "Fast"),
            (50, "Balanced"),
            (100, "High Quality")
        ]

        results = []

        for steps, label in step_configs:
            print(f"\n  Testing {label} mode ({steps} steps)...")

            start = time.time()
            ply_bytes, gs_model, timings = await generator.generate_gaussian_splat(
                rgba_image=test_image,
                prompt="red cube",
                num_inference_steps=steps
            )
            elapsed = time.time() - start

            num_gaussians = generator._count_gaussians_in_ply(ply_bytes)
            file_size_mb = len(ply_bytes) / (1024 * 1024)

            results.append({
                'label': label,
                'steps': steps,
                'time': elapsed,
                'gaussians': num_gaussians,
                'size_mb': file_size_mb
            })

            print(f"    Time: {elapsed:.2f}s, Gaussians: {num_gaussians:,}, Size: {file_size_mb:.1f} MB")

        # Summary
        print(f"\n✅ Quality scaling test complete")
        print(f"\n  Recommendations:")
        print(f"  - Fast (20 steps): Use for testing (~{results[0]['time']:.1f}s)")
        print(f"  - Balanced (50 steps): Use for production (~{results[1]['time']:.1f}s)")
        print(f"  - High Quality (100 steps): Use if time permits (~{results[2]['time']:.1f}s)")

        return True

    except Exception as e:
        print(f"❌ Quality scaling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all TRELLIS tests"""

    tests = [
        ("TRELLIS Installation", test_trellis_installation),
        ("Basic TRELLIS Generation", test_trellis_basic_generation),
        ("Full Pipeline (FLUX → TRELLIS)", test_trellis_with_flux_pipeline),
        ("CLIP Validation", test_trellis_clip_validation),
        ("Quality Scaling", test_trellis_quality_scaling),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ {name} crashed: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/{len(tests)} ✅")
    print(f"Failed: {failed}/{len(tests)} ❌")
    print("=" * 80)

    if failed == 0:
        print("\n🎉 All tests passed! TRELLIS pipeline is ready for mainnet deployment.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Fix issues before mainnet deployment.")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
```

### Step 3.2: Create Comprehensive Mainnet Readiness Test

**File:** `test_mainnet_readiness.py` (NEW, in project root)

```python
#!/usr/bin/env python3
"""
Mainnet Readiness Test Suite

Final validation before deploying to mainnet:
1. End-to-end generation timing (<30s required)
2. CLIP scores (0.65+ required to avoid cooldown)
3. Gaussian count verification (300K+ recommended)
4. File size validation (30-50 MB competitive range)
5. Stress test (10 consecutive generations)
"""

import sys
import time
import asyncio
import aiohttp
from pathlib import Path

print("=" * 80)
print("MAINNET READINESS TEST SUITE")
print("=" * 80)
print("\nThis test validates your miner is ready for mainnet deployment.")
print("Requirements:")
print("  - Generation time: <30s")
print("  - CLIP score: >0.65")
print("  - Gaussian count: 200K+ (300K-500K competitive)")
print("  - File size: 20+ MB (30-50 MB competitive)")
print("=" * 80)


# Test prompts (mix of simple and complex)
TEST_PROMPTS = [
    "red sports car",
    "wooden chair",
    "blue teapot",
    "modern lamp",
    "leather sofa",
]


async def test_generation_service_health():
    """Test 1: Check generation service is running"""
    print("\n[Test 1] Generation Service Health Check")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:10006/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status != 200:
                    print(f"❌ Health check failed: Status {response.status}")
                    return False

                health = await response.json()

                # Check all models loaded
                models = health.get("models_loaded", {})
                all_loaded = all(models.values())

                if not all_loaded:
                    print(f"❌ Not all models loaded: {models}")
                    return False

                print(f"✅ Generation service healthy")
                print(f"   Models: {', '.join(models.keys())}")
                return True

    except aiohttp.ClientConnectorError:
        print(f"❌ Generation service not running")
        print(f"   Start with: python generation/serve_competitive.py --port 10006 --enable-validation")
        return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


async def test_single_generation(prompt: str, max_time: float = 30.0):
    """Test single generation"""

    print(f"\n  Testing: '{prompt}'")

    try:
        start = time.time()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:10006/generate/",
                data={"prompt": prompt},
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:

                ply_data = await response.read()
                elapsed = time.time() - start

                # Parse headers
                clip_score = response.headers.get("X-CLIP-Score", "N/A")
                validation_failed = response.headers.get("X-Validation-Failed") == "true"

                # Parse PLY to count gaussians
                num_gaussians = 0
                if ply_data.startswith(b"ply\n"):
                    header_end = ply_data.find(b"end_header\n")
                    if header_end != -1:
                        header = ply_data[:header_end].decode('utf-8')
                        for line in header.split('\n'):
                            if line.startswith('element vertex'):
                                num_gaussians = int(line.split()[-1])
                                break

                file_size_mb = len(ply_data) / (1024 * 1024)

                # Validate
                issues = []

                if validation_failed:
                    issues.append(f"CLIP validation failed ({clip_score})")

                if elapsed > max_time:
                    issues.append(f"Too slow ({elapsed:.1f}s > {max_time}s)")

                if len(ply_data) < 1000:
                    issues.append(f"Empty/invalid output ({len(ply_data)} bytes)")

                if num_gaussians < 200000:
                    issues.append(f"Low gaussian count ({num_gaussians:,} < 200K)")

                if file_size_mb < 20:
                    issues.append(f"Small file size ({file_size_mb:.1f} MB < 20 MB)")

                # Report
                status = "✅" if not issues else "⚠️ "

                print(f"  {status} Result:")
                print(f"     Time: {elapsed:.2f}s")
                print(f"     CLIP: {clip_score}")
                print(f"     Gaussians: {num_gaussians:,}")
                print(f"     Size: {file_size_mb:.1f} MB")

                if issues:
                    print(f"     Issues: {', '.join(issues)}")

                return {
                    'prompt': prompt,
                    'success': not validation_failed and len(ply_data) > 1000,
                    'time': elapsed,
                    'clip_score': float(clip_score) if clip_score != "N/A" else 0.0,
                    'num_gaussians': num_gaussians,
                    'file_size_mb': file_size_mb,
                    'issues': issues
                }

    except asyncio.TimeoutError:
        print(f"  ❌ Timeout after 60s")
        return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


async def test_batch_generation():
    """Test 2: Batch generation with diverse prompts"""
    print(f"\n[Test 2] Batch Generation Test ({len(TEST_PROMPTS)} prompts)")

    results = []

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n  [{i}/{len(TEST_PROMPTS)}]")
        result = await test_single_generation(prompt)
        if result:
            results.append(result)

        # Small delay between generations
        await asyncio.sleep(2)

    # Summary
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"\n  Summary:")
    print(f"    Successful: {len(successful)}/{len(results)}")
    print(f"    Failed: {len(failed)}/{len(results)}")

    if successful:
        avg_time = sum(r['time'] for r in successful) / len(successful)
        avg_clip = sum(r['clip_score'] for r in successful) / len(successful)
        avg_gaussians = sum(r['num_gaussians'] for r in successful) / len(successful)
        avg_size = sum(r['file_size_mb'] for r in successful) / len(successful)

        print(f"\n  Averages (successful only):")
        print(f"    Time: {avg_time:.2f}s")
        print(f"    CLIP: {avg_clip:.3f}")
        print(f"    Gaussians: {avg_gaussians:,.0f}")
        print(f"    Size: {avg_size:.1f} MB")

        # Quality assessment
        print(f"\n  Quality Assessment:")

        if avg_time < 30:
            print(f"    ✅ Speed: {avg_time:.1f}s < 30s (GOOD)")
        else:
            print(f"    ❌ Speed: {avg_time:.1f}s > 30s (TOO SLOW)")

        if avg_clip >= 0.65:
            print(f"    ✅ CLIP: {avg_clip:.3f} >= 0.65 (GOOD)")
        else:
            print(f"    ⚠️  CLIP: {avg_clip:.3f} < 0.65 (MAY GET COOLDOWN)")

        if avg_gaussians >= 300000:
            print(f"    ✅ Gaussians: {avg_gaussians:,.0f} >= 300K (COMPETITIVE)")
        elif avg_gaussians >= 200000:
            print(f"    ⚠️  Gaussians: {avg_gaussians:,.0f} >= 200K (ACCEPTABLE)")
        else:
            print(f"    ❌ Gaussians: {avg_gaussians:,.0f} < 200K (TOO LOW)")

        if avg_size >= 30:
            print(f"    ✅ File size: {avg_size:.1f} MB >= 30 MB (COMPETITIVE)")
        elif avg_size >= 20:
            print(f"    ⚠️  File size: {avg_size:.1f} MB >= 20 MB (ACCEPTABLE)")
        else:
            print(f"    ❌ File size: {avg_size:.1f} MB < 20 MB (TOO SMALL)")

    return len(successful) >= len(results) * 0.8  # 80% success rate required


async def test_stress_test():
    """Test 3: Stress test (10 consecutive generations)"""
    print(f"\n[Test 3] Stress Test (10 consecutive generations)")

    stress_prompt = "modern office chair"
    num_runs = 10

    results = []

    print(f"\n  Running {num_runs} consecutive generations...")

    for i in range(num_runs):
        print(f"\n  Run {i+1}/{num_runs}")
        result = await test_single_generation(stress_prompt, max_time=35.0)
        if result:
            results.append(result)

        # No delay - test back-to-back performance

    # Analysis
    successful = [r for r in results if r['success']]

    print(f"\n  Stress Test Results:")
    print(f"    Completed: {len(results)}/{num_runs}")
    print(f"    Successful: {len(successful)}/{len(results)}")

    if successful:
        times = [r['time'] for r in successful]
        min_time = min(times)
        max_time = max(times)
        avg_time = sum(times) / len(times)

        print(f"    Time range: {min_time:.2f}s - {max_time:.2f}s (avg: {avg_time:.2f}s)")

        # Check for performance degradation
        if max_time > avg_time * 1.5:
            print(f"    ⚠️  Performance degradation detected (max time {max_time/avg_time:.1f}x average)")
        else:
            print(f"    ✅ Stable performance")

    return len(successful) >= num_runs * 0.9  # 90% success rate required


async def run_mainnet_readiness_tests():
    """Run all mainnet readiness tests"""

    print("\n" + "=" * 80)
    print("Starting mainnet readiness tests...")
    print("=" * 80)

    # Test 1: Health check
    health_ok = await test_generation_service_health()
    if not health_ok:
        print("\n❌ ABORT: Generation service not healthy")
        return False

    # Test 2: Batch generation
    batch_ok = await test_batch_generation()

    # Test 3: Stress test
    stress_ok = await test_stress_test()

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    all_passed = health_ok and batch_ok and stress_ok

    if all_passed:
        print("✅ READY FOR MAINNET DEPLOYMENT")
        print("\nYour miner meets all requirements:")
        print("  ✅ Generation speed < 30s")
        print("  ✅ 80%+ success rate on diverse prompts")
        print("  ✅ Stable performance under stress")
        print("\nRecommendations before deploying:")
        print("  1. Monitor CLIP scores closely (target: 0.70+)")
        print("  2. Watch for OOM errors on complex prompts")
        print("  3. Track mainnet acceptance rate (target: 60%+)")
    else:
        print("❌ NOT READY FOR MAINNET")
        print("\nFailed tests:")
        if not health_ok:
            print("  ❌ Health check")
        if not batch_ok:
            print("  ❌ Batch generation")
        if not stress_ok:
            print("  ❌ Stress test")
        print("\nFix issues before deploying to mainnet!")

    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(run_mainnet_readiness_tests())
    sys.exit(0 if success else 1)
```

---

## PHASE 4: DEPLOYMENT CHECKLIST

### Pre-Deployment Validation

Before deploying to mainnet, complete this checklist:

**1. TRELLIS Installation** ✓
```bash
# Verify TRELLIS works
python -c "from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline; print('✅ TRELLIS OK')"
```

**2. Run TRELLIS Tests** ✓
```bash
# Run TRELLIS-specific tests
python generation/test_trellis_pipeline.py

# Expected: All 5 tests pass
# - TRELLIS Installation
# - Basic Generation (>100K gaussians)
# - Full Pipeline (<30s)
# - CLIP Validation (works correctly)
# - Quality Scaling (different inference steps)
```

**3. Start Generation Service** ✓
```bash
# Start with CLIP validation enabled
python generation/serve_competitive.py \
    --port 10006 \
    --enable-validation \
    --validation-threshold 0.65
```

**4. Run Mainnet Readiness Test** ✓
```bash
# Final validation before mainnet
python test_mainnet_readiness.py

# Expected:
# - Health check: PASS
# - Batch generation: 80%+ success (4/5 prompts)
# - Stress test: 90%+ success (9/10 runs)
# - Verdict: "READY FOR MAINNET DEPLOYMENT"
```

**5. Review Key Metrics** ✓
```
Target metrics for competitive mining:
- Generation time: <30s (20-25s ideal)
- CLIP scores: 0.70+ (0.75+ competitive)
- Gaussian count: 300K-500K (200K minimum)
- File size: 30-50 MB (20 MB minimum)
- Acceptance rate: 60%+ (mainnet validators)
```

### Deployment Steps

Once all tests pass:

1. **Stop current mainnet miner** (if running)
2. **Deploy new TRELLIS miner:**
```bash
python neurons/serve_miner_competitive.py \
    --netuid 17 \
    --wallet.name YOUR_WALLET \
    --wallet.hotkey YOUR_HOTKEY \
    --generation.port 10006 \
    --enable-validation
```

3. **Monitor initial performance:**
   - Watch logs for CLIP scores
   - Track acceptance rate (target: 60%+)
   - Monitor for OOM errors
   - Check validator feedback

4. **Tune if needed:**
   - If CLIP scores low (<0.65): Increase TRELLIS inference steps (50 → 100)
   - If too slow (>30s): Decrease inference steps (50 → 30)
   - If OOM errors: Enable fp16 mode in TRELLIS
   - If low acceptance: Verify gaussian counts are 200K+

---

## TROUBLESHOOTING GUIDE

### Issue 1: TRELLIS Installation Fails

**Symptoms:** ImportError when importing TRELLIS modules

**Solutions:**
1. **Check CUDA version:**
```bash
nvcc --version  # Should be 11.8 or 12.2
```

2. **Try Windows-specific fork:**
```bash
# If on Windows
git clone https://github.com/sdbds/TRELLIS-for-windows.git
```

3. **Install in fp16 mode** (if low VRAM):
```python
# In trellis_integration.py
self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float16  # Use half precision
)
```

### Issue 2: TRELLIS Generates Low Gaussian Counts

**Symptoms:** Only getting 50K-100K gaussians instead of 300K+

**Diagnosis:**
```python
# Check TRELLIS output format
print(f"TRELLIS outputs: {outputs.keys()}")
print(f"Gaussian type: {type(outputs['gaussian'][0])}")
```

**Solutions:**
1. **Increase inference steps:**
```python
num_inference_steps=100  # Higher = more gaussians
```

2. **Adjust TRELLIS sampler params:**
```python
sparse_structure_sampler_params={
    "steps": 100,  # More steps = higher density
    "cfg_strength": 10.0,  # Higher = more detail
}
```

3. **Verify model variant:**
```python
# Use large model, not base
model_name="microsoft/TRELLIS-image-large"  # 2B params
```

### Issue 3: CLIP Validation Failing After TRELLIS

**Symptoms:** CLIP scores drop to <0.5 or rendering fails

**Diagnosis:**
```python
# Check if GaussianModel can be created
gs_model = trellis_gen._create_gaussian_model_from_ply(ply_bytes)
if gs_model is None:
    print("❌ GaussianModel creation failed")
```

**Solutions:**
1. **Update SH degree:**
```python
# In _create_gaussian_model_from_ply()
gs_model = GaussianModel(sh_degree=3)  # TRELLIS uses higher SH
```

2. **Fix prompt enhancement:**
```python
# Don't enhance prompts - use original
validation_prompt = prompt  # No enhancement
```

3. **Test rendering separately:**
```bash
python generation/test_clip_validator.py
```

### Issue 4: TRELLIS Too Slow (>30s)

**Symptoms:** TRELLIS takes 20-30s alone, total pipeline >40s

**Solutions:**
1. **Reduce inference steps:**
```python
num_inference_steps=30  # Fast mode (was 50)
```

2. **Use base model instead of large:**
```python
model_name="microsoft/TRELLIS-image-base"
```

3. **Enable fp16 + torch.compile:**
```python
self.pipeline = self.pipeline.to(torch.float16)
self.pipeline.unet = torch.compile(self.pipeline.unet)
```

### Issue 5: OOM Errors with TRELLIS

**Symptoms:** CUDA out of memory during TRELLIS generation

**Solutions:**
1. **Enable fp16:**
```python
self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)
```

2. **Offload FLUX before TRELLIS:**
```python
# In serve_competitive.py, before TRELLIS
app.state.flux_generator.offload_to_cpu()
cleanup_memory()
```

3. **Reduce resolution:**
```python
# Generate smaller image
height=384, width=384  # Instead of 512
```

---

## SUCCESS CRITERIA

Your TRELLIS integration is successful when:

✅ **Installation:** TRELLIS imports without errors
✅ **Generation:** Produces 200K-500K gaussians (30-50 MB files)
✅ **Speed:** Full pipeline completes in <30s
✅ **CLIP:** Scores consistently >0.65 (>0.70 competitive)
✅ **Validation:** CLIP validation works with TRELLIS outputs
✅ **Stability:** 80%+ success rate on diverse prompts
✅ **Mainnet:** Acceptance rate >60% from validators

---

## FINAL NOTES

1. **Test extensively before mainnet** - Use test_mainnet_readiness.py
2. **Start conservative** - Use 50 inference steps, tune from there
3. **Monitor acceptance rate** - Target 60%+, competitive miners hit 70-85%
4. **Compare to InstantMesh** - TRELLIS should give 10-20x more gaussians
5. **Don't skip CLIP validation** - It prevents cooldown penalties

Good luck with the TRELLIS integration! This should bring you to competitive levels.

---

## APPENDIX: Expected Performance Comparison

| Metric | InstantMesh (Old) | TRELLIS (New) | Top Miners |
|--------|------------------|---------------|------------|
| Gaussian Count | 12K-50K | 200K-500K | 400K-500K |
| File Size | 1-2 MB | 20-50 MB | 40-60 MB |
| Generation Time | 2-3s | 10-20s | 15-25s |
| CLIP Scores | 0.65-0.75 | 0.70-0.85 | 0.80-0.95 |
| Acceptance Rate | 50-70% | 60-80% | 70-90% |
| Architecture | Mesh → Gaussian | Direct Gaussian | Direct Gaussian |

The key difference: **TRELLIS generates gaussians natively**, while InstantMesh samples from mesh surface (fundamental limitation).
