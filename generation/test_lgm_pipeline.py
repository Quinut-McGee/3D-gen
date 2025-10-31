"""
Test LGM Pipeline Standalone

This script tests the LGM gaussian generation pipeline independently
to verify it works before integrating into serve_competitive.py
"""

import asyncio
import time
from PIL import Image
import numpy as np
from loguru import logger
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation.lgm_integration import LGMGaussianGenerator


def create_test_image():
    """Create a simple test RGBA image"""
    # Create a 512x512 image with a white circle on transparent background
    size = 512
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    pixels = img.load()

    center = size // 2
    radius = size // 3

    for x in range(size):
        for y in range(size):
            dist = ((x - center) ** 2 + (y - center) ** 2) ** 0.5
            if dist < radius:
                # White circle
                pixels[x, y] = (255, 255, 255, 255)

    return img


async def test_lgm_generation():
    """Test LGM generation end-to-end"""
    logger.info("=" * 60)
    logger.info("🧪 TESTING LGM PIPELINE STANDALONE")
    logger.info("=" * 60)

    try:
        # Initialize LGM generator
        logger.info("\n[1/4] Initializing LGM generator...")
        start_init = time.time()

        lgm_generator = LGMGaussianGenerator(device="cuda")

        init_time = time.time() - start_init
        logger.info(f"✅ LGM initialized in {init_time:.1f}s")

        # Create test image
        logger.info("\n[2/4] Creating test image...")
        test_image = create_test_image()
        logger.info(f"✅ Test image created: {test_image.size}, mode: {test_image.mode}")

        # Generate gaussians
        logger.info("\n[3/4] Generating Gaussian Splat with LGM...")
        start_gen = time.time()

        ply_bytes, gs_model, timings = await lgm_generator.generate_gaussian_splat(
            rgba_image=test_image,
            prompt="test image",  # Prompt not used by LGM but kept for consistency
            guidance_scale=5.0,
            num_inference_steps=30
        )

        gen_time = time.time() - start_gen

        # Analyze results
        logger.info("\n[4/4] Analyzing results...")

        file_size_mb = len(ply_bytes) / (1024 * 1024)

        # Count gaussians from PLY header
        header_end = ply_bytes.find(b"end_header\n")
        header = ply_bytes[:header_end].decode('utf-8')
        num_gaussians = 0
        for line in header.split('\n'):
            if line.startswith('element vertex'):
                num_gaussians = int(line.split()[-1])
                break

        logger.info("=" * 60)
        logger.info("📊 RESULTS")
        logger.info("=" * 60)
        logger.info(f"Generation time: {gen_time:.2f}s")
        logger.info(f"File size: {file_size_mb:.1f} MB")
        logger.info(f"Gaussian count: {num_gaussians:,}")
        logger.info(f"LGM inference: {timings.get('lgm', 0):.2f}s")

        # Quality assessment
        logger.info("\n" + "=" * 60)
        logger.info("✅ QUALITY ASSESSMENT")
        logger.info("=" * 60)

        passed = True

        # Check generation time (<20s target)
        if gen_time < 20:
            logger.info(f"✅ Generation time: {gen_time:.2f}s < 20s (PASS)")
        else:
            logger.warning(f"⚠️  Generation time: {gen_time:.2f}s >= 20s (MARGINAL)")
            passed = False

        # Check file size (>10 MB target)
        if file_size_mb >= 10:
            logger.info(f"✅ File size: {file_size_mb:.1f} MB >= 10 MB (PASS)")
        else:
            logger.warning(f"❌ File size: {file_size_mb:.1f} MB < 10 MB (FAIL)")
            passed = False

        # Check gaussian count (>100K target)
        if num_gaussians >= 100000:
            logger.info(f"✅ Gaussian count: {num_gaussians:,} >= 100K (PASS)")
        else:
            logger.warning(f"❌ Gaussian count: {num_gaussians:,} < 100K (FAIL)")
            passed = False

        # Overall verdict
        logger.info("\n" + "=" * 60)
        if passed:
            logger.info("✅✅✅ LGM PIPELINE TEST: PASSED ✅✅✅")
            logger.info("Ready for integration into serve_competitive.py")
        else:
            logger.warning("⚠️⚠️⚠️ LGM PIPELINE TEST: MARGINAL ⚠️⚠️⚠️")
            logger.warning("Review results before integration")
        logger.info("=" * 60)

        return passed

    except Exception as e:
        logger.error(f"❌ LGM TEST FAILED: {e}", exc_info=True)
        logger.info("\n" + "=" * 60)
        logger.error("❌❌❌ LGM PIPELINE TEST: FAILED ❌❌❌")
        logger.info("=" * 60)
        return False


if __name__ == "__main__":
    logger.info("Starting LGM pipeline test...")

    # Run test
    result = asyncio.run(test_lgm_generation())

    # Exit with appropriate code
    sys.exit(0 if result else 1)
