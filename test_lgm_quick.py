#!/usr/bin/env python3
"""
Quick LGM test before hot swap
Verifies LGM can generate a gaussian splat
"""
import sys
sys.path.insert(0, '/home/kobe/404-gen/v1/3D-gen/generation')

import asyncio
from PIL import Image
import numpy as np
from loguru import logger
from lgm_integration import LGMGaussianGenerator

async def test_lgm():
    print("=" * 70)
    print("LGM QUICK GENERATION TEST")
    print("=" * 70)

    # Create test RGBA image (white square on transparent background)
    print("\n[1/4] Creating test RGBA image...")
    img = np.zeros((512, 512, 4), dtype=np.uint8)
    img[128:384, 128:384, :3] = 255  # White square
    img[128:384, 128:384, 3] = 255   # Opaque
    rgba_image = Image.fromarray(img, 'RGBA')
    print("  ✅ Test image created: 512x512 RGBA")

    # Initialize LGM
    print("\n[2/4] Loading LGM pipeline...")
    try:
        lgm_gen = LGMGaussianGenerator(device="cuda")
        print("  ✅ LGM pipeline loaded")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Generate gaussian splat
    print("\n[3/4] Generating Gaussian Splat with LGM...")
    try:
        ply_bytes, gs_model, timings = await lgm_gen.generate_gaussian_splat(
            rgba_image=rgba_image,
            prompt="test cube",
            guidance_scale=5.0,
            num_inference_steps=30
        )

        num_gaussians = len(ply_bytes) // 43  # Rough estimate
        file_size_mb = len(ply_bytes) / (1024 * 1024)

        print(f"  ✅ Generation successful!")
        print(f"     Time: {timings['total_3d']:.2f}s")
        print(f"     Size: {file_size_mb:.1f} MB")
        print(f"     Gaussians: ~{num_gaussians:,}")

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify PLY format
    print("\n[4/4] Verifying PLY compatibility...")
    try:
        from DreamGaussianLib.GaussianSplattingModel import GaussianModel
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp:
            tmp.write(ply_bytes)
            tmp_path = tmp.name

        gs = GaussianModel(3)
        gs.load_ply(tmp_path)
        os.unlink(tmp_path)

        print(f"  ✅ PLY format compatible!")
        print(f"     Loaded {len(gs._xyz):,} gaussians")
        print(f"     Opacity mean: {gs._opacity.mean():.4f}")
        print(f"     Opacity std: {gs._opacity.std():.4f}")

        if gs._opacity.std() < 0.1:
            print("  ⚠️  WARNING: Low opacity variance detected")
            return False

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("✅✅✅ LGM TEST PASSED - READY FOR HOT SWAP! ✅✅✅")
    print("=" * 70)
    return True

if __name__ == "__main__":
    result = asyncio.run(test_lgm())
    sys.exit(0 if result else 1)
