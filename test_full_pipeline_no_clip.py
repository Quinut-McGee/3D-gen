"""
Test Phase 2: Full Pipeline Without CLIP Validation
Test: FLUX → RMBG → TRELLIS (no CLIP validation)
"""

import asyncio
import httpx
import time
from loguru import logger

async def test_full_pipeline():
    """Test full pipeline without CLIP validation"""

    logger.info("=" * 60)
    logger.info("TEST PHASE 2: Full Pipeline (No CLIP Validation)")
    logger.info("=" * 60)

    test_prompt = "a futuristic sports car, red paint, sleek design"
    logger.info(f"\nTest prompt: \"{test_prompt}\"")

    # Call generation service
    logger.info("\n[1/1] Calling generation service...")
    url = "http://localhost:8080/generate/"

    t_start = time.time()

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                url,
                data={
                    "prompt": test_prompt
                }
            )

        t_end = time.time()
        t_total = t_end - t_start

        if response.status_code != 200:
            logger.error(f"  ❌ Generation failed with status {response.status_code}")
            logger.error(f"     Response: {response.text}")
            return False

        result = response.json()

        if not result.get("success", False):
            logger.error(f"  ❌ Generation failed: {result.get('error', 'Unknown error')}")
            return False

        logger.info(f"  ✅ Generation complete ({t_total:.2f}s)")

    except httpx.TimeoutException:
        logger.error(f"  ❌ Timeout after 120s")
        return False
    except Exception as e:
        logger.error(f"  ❌ Generation failed: {e}")
        return False

    # Analyze results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS:")
    logger.info("=" * 60)

    logger.info(f"Total time: {t_total:.2f}s")
    logger.info(f"Gaussian count: {result.get('num_gaussians', 'N/A'):,}")
    logger.info(f"File size: {result.get('file_size_mb', 'N/A'):.1f} MB")

    # Check targets
    success = True

    if t_total > 20.0:
        logger.warning(f"⚠️  Total time {t_total:.2f}s > 20s target (we want buffer under 30s limit)")
        success = False
    else:
        logger.info(f"✅ Total time {t_total:.2f}s < 20s target")

    num_gaussians = result.get('num_gaussians', 0)
    if num_gaussians < 200000:
        logger.warning(f"⚠️  Gaussian count {num_gaussians:,} < 200K target")
        success = False
    else:
        logger.info(f"✅ Gaussian count {num_gaussians:,} > 200K target")

    file_size_mb = result.get('file_size_mb', 0)
    if file_size_mb < 15.0:
        logger.warning(f"⚠️  File size {file_size_mb:.1f} MB < 15 MB target")
        success = False
    else:
        logger.info(f"✅ File size {file_size_mb:.1f} MB > 15 MB target")

    logger.info("\n" + "=" * 60)
    if success:
        logger.info("✅ TEST PHASE 2 PASSED - Full pipeline working!")
    else:
        logger.error("❌ TEST PHASE 2 FAILED - Check warnings above")
    logger.info("=" * 60)

    return success

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
