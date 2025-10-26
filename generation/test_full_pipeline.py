#!/usr/bin/env python3
"""
End-to-end test of the competitive generation pipeline.

Tests:
1. Generation timing (target: <30s)
2. CLIP score analysis (target: >0.6)
3. 2D vs 3D bottleneck identification

Usage:
    python test_full_pipeline.py
"""
import time
import requests
from PIL import Image
import io

print("=" * 80)
print("END-TO-END GENERATION PIPELINE TEST")
print("=" * 80)

# Test prompts (varied complexity)
test_prompts = [
    "red sports car",
    "blue teapot",
    "wooden chair"
]

GENERATION_URL = "http://localhost:8093/generate/"

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{'=' * 80}")
    print(f"TEST {i}/{len(test_prompts)}: '{prompt}'")
    print(f"{'=' * 80}")

    # Generate
    print(f"\n[1/2] Calling generation service...")
    start_time = time.time()

    try:
        response = requests.post(
            GENERATION_URL,
            data={"prompt": prompt},
            timeout=90  # 90s timeout
        )

        generation_time = time.time() - start_time

        if response.status_code == 200:
            result_size = len(response.content)
            clip_score = response.headers.get("X-CLIP-Score", "N/A")
            validation_failed = response.headers.get("X-Validation-Failed") == "true"

            print(f"✅ Generation completed in {generation_time:.1f}s")
            print(f"   Result size: {result_size / 1024:.1f} KB")
            print(f"   CLIP score: {clip_score}")
            print(f"   Validation: {'❌ FAILED' if validation_failed else '✅ PASSED'}")

            # Save PLY
            output_path = f"/tmp/test_pipeline_{i}_{prompt.replace(' ', '_')}.ply"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"   Saved to: {output_path}")

            # Analyze timing
            print(f"\n[2/2] Performance analysis:")
            if generation_time < 30:
                print(f"   ✅ Speed GOOD: {generation_time:.1f}s < 30s target")
            else:
                print(f"   ❌ Speed SLOW: {generation_time:.1f}s > 30s target (need {30 - generation_time:.1f}s improvement)")

            # Analyze CLIP
            try:
                clip_float = float(clip_score)
                if clip_float >= 0.6:
                    print(f"   ✅ CLIP GOOD: {clip_float:.3f} >= 0.6 target")
                else:
                    print(f"   ❌ CLIP LOW: {clip_float:.3f} < 0.6 target (need +{0.6 - clip_float:.3f} improvement)")
            except ValueError:
                print(f"   ⚠️  CLIP score not available")

        else:
            print(f"❌ Generation failed with status {response.status_code}")
            print(f"   Response: {response.text[:200]}")

    except requests.Timeout:
        print(f"❌ Generation timeout after 90s")
    except Exception as e:
        print(f"❌ Error: {e}")

print(f"\n{'=' * 80}")
print("TEST COMPLETE")
print(f"{'=' * 80}")
print("\nNext steps:")
print("1. Check /tmp/debug_* files to see intermediate stages")
print("2. If CLIP scores are low (<0.6), problem is likely 2D image generation")
print("3. If timing is slow (>30s), check which stage takes longest")
print("4. Review generation-competitive logs: pm2 logs generation-competitive")
