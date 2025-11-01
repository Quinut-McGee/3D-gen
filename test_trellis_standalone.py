"""
Test Phase 1: TRELLIS Microservice Standalone Test
Test the TRELLIS microservice directly with a simple image.
"""

import io
import base64
import httpx
import time
from PIL import Image
from loguru import logger

def test_trellis_microservice():
    """Test TRELLIS microservice with a simple test image"""

    logger.info("=" * 60)
    logger.info("TEST PHASE 1: TRELLIS Microservice Standalone")
    logger.info("=" * 60)

    # Create a simple test image (white square on black background)
    logger.info("\n[1/3] Creating test image...")
    test_image = Image.new('RGB', (512, 512), color='black')
    # Draw a white square in the center
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_image)
    draw.rectangle([156, 156, 356, 356], fill='white')
    logger.info("  ✅ Test image created (512x512, white square on black)")

    # Convert to base64
    logger.info("\n[2/3] Calling TRELLIS microservice...")
    buffer = io.BytesIO()
    test_image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Call TRELLIS
    trellis_url = "http://localhost:10008"
    t_start = time.time()

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{trellis_url}/generate",
                json={
                    "image_base64": image_base64,
                    "seed": 42,
                    "timeout": 30
                }
            )

        t_end = time.time()

        if response.status_code != 200:
            logger.error(f"  ❌ TRELLIS returned status {response.status_code}")
            logger.error(f"     Response: {response.text}")
            return False

        result = response.json()

        if not result.get("success", False):
            logger.error(f"  ❌ TRELLIS generation failed: {result.get('error', 'Unknown error')}")
            return False

        logger.info(f"  ✅ TRELLIS responded ({t_end-t_start:.2f}s)")

    except Exception as e:
        logger.error(f"  ❌ TRELLIS microservice call failed: {e}")
        return False

    # Verify results
    logger.info("\n[3/3] Verifying results...")

    ply_bytes = base64.b64decode(result["ply_base64"])
    num_gaussians = result["num_gaussians"]
    file_size_mb = result["file_size_mb"]
    generation_time = result["generation_time"]

    logger.info(f"  Generation time: {generation_time:.2f}s")
    logger.info(f"  Gaussians: {num_gaussians:,}")
    logger.info(f"  File size: {file_size_mb:.1f} MB")

    # Check targets
    success = True
    if generation_time > 10.0:
        logger.warning(f"  ⚠️  Generation time {generation_time:.2f}s > 10s target")
        success = False
    else:
        logger.info(f"  ✅ Generation time {generation_time:.2f}s < 10s target")

    if num_gaussians < 150000:
        logger.warning(f"  ⚠️  Gaussian count {num_gaussians:,} < 150K target")
        success = False
    else:
        logger.info(f"  ✅ Gaussian count {num_gaussians:,} > 150K target")

    if file_size_mb < 10.0:
        logger.warning(f"  ⚠️  File size {file_size_mb:.1f} MB < 10 MB target")
        success = False
    else:
        logger.info(f"  ✅ File size {file_size_mb:.1f} MB > 10 MB target")

    logger.info("\n" + "=" * 60)
    if success:
        logger.info("✅ TEST PHASE 1 PASSED - TRELLIS microservice working!")
    else:
        logger.error("❌ TEST PHASE 1 FAILED - Check warnings above")
    logger.info("=" * 60)

    return success

if __name__ == "__main__":
    test_trellis_microservice()
