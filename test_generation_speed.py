#!/usr/bin/env python3
"""
Speed test script for 404-GEN miner generation service.
Tests different configs to find optimal speed/quality balance.
"""

import time
import requests
import sys
from pathlib import Path

# Test prompts representing different complexity levels
TEST_PROMPTS = [
    "a red sports car",  # Simple
    "a wooden chair with metal legs",  # Medium
    "a fantasy medieval castle with towers",  # Complex
]

CONFIGS_TO_TEST = [
    ("text_mv_ultra_fast.yaml", "Ultra-Fast (<15s target)"),
    ("text_mv_fast.yaml", "Fast (<25s target)"),
    ("text_mv.yaml", "Quality (~3.5min)"),
]

ENDPOINT = "http://localhost:10006/generate/"


def test_generation(prompt: str, timeout: int = 300) -> tuple[bool, float, int]:
    """
    Test a single generation.
    Returns: (success, time_taken, file_size)
    """
    print(f"\n  Testing prompt: '{prompt}'")

    start_time = time.time()

    try:
        response = requests.post(
            ENDPOINT,
            data={"prompt": prompt},
            timeout=timeout
        )

        elapsed = time.time() - start_time

        if response.status_code == 200:
            file_size = len(response.content)
            print(f"    ✅ Success in {elapsed:.2f}s ({file_size/1024:.1f} KB)")
            return True, elapsed, file_size
        else:
            print(f"    ❌ Failed with status {response.status_code}")
            return False, elapsed, 0

    except requests.Timeout:
        elapsed = time.time() - start_time
        print(f"    ⏱️  Timeout after {elapsed:.2f}s")
        return False, elapsed, 0

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"    ❌ Error: {e}")
        return False, elapsed, 0


def test_config(config_name: str, description: str):
    """Test a specific configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Config: {config_name}")
    print(f"{'='*60}")

    results = []

    for prompt in TEST_PROMPTS:
        success, time_taken, file_size = test_generation(prompt)
        results.append({
            "prompt": prompt,
            "success": success,
            "time": time_taken,
            "size": file_size
        })

        # Brief pause between tests
        time.sleep(2)

    # Summary
    successful = [r for r in results if r["success"]]

    if successful:
        avg_time = sum(r["time"] for r in successful) / len(successful)
        avg_size = sum(r["size"] for r in successful) / len(successful)

        print(f"\n📊 Summary for {config_name}:")
        print(f"  Success rate: {len(successful)}/{len(results)}")
        print(f"  Average time: {avg_time:.2f}s")
        print(f"  Average size: {avg_size/1024:.1f} KB")

        # Check if meets requirements
        if avg_time < 30:
            print(f"  ✅ MEETS 300s cooldown requirement (<30s)")
        elif avg_time < 120:
            print(f"  ⚠️  May work with 120s cooldown (UID49)")
        else:
            print(f"  ❌ TOO SLOW for production")

        if avg_size > 1000:
            print(f"  ✅ File size acceptable (>{1000}B)")
        else:
            print(f"  ⚠️  File size may be too small")
    else:
        print(f"\n❌ All tests failed for {config_name}")

    return results


def main():
    print("="*60)
    print("404-GEN Miner Speed Test")
    print("="*60)
    print(f"\nEndpoint: {ENDPOINT}")
    print(f"Test prompts: {len(TEST_PROMPTS)}")
    print(f"Configs to test: {len(CONFIGS_TO_TEST)}")

    # Check if service is running
    try:
        response = requests.get("http://localhost:10006/", timeout=2)
        print("✅ Generation service is running")
    except Exception:
        print("❌ Generation service not accessible!")
        print("   Start it with: cd generation && python serve.py")
        sys.exit(1)

    # Test each config
    all_results = {}

    for config_file, description in CONFIGS_TO_TEST:
        # Note: This test assumes you manually restart serve.py with different configs
        # Or you can modify serve.py to accept runtime config changes

        print(f"\n\n{'#'*60}")
        print(f"MANUAL STEP REQUIRED:")
        print(f"1. Stop the current generation service (Ctrl+C or pm2 stop generation)")
        print(f"2. Start with: python serve.py --config configs/{config_file}")
        print(f"3. Press ENTER when ready to test...")
        print(f"{'#'*60}")

        input()

        results = test_config(config_file, description)
        all_results[config_file] = results

    # Final comparison
    print(f"\n\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}\n")

    print(f"{'Config':<30} {'Avg Time':<12} {'Avg Size':<12} {'Status'}")
    print("-" * 70)

    for config_file, description in CONFIGS_TO_TEST:
        if config_file in all_results:
            results = all_results[config_file]
            successful = [r for r in results if r["success"]]

            if successful:
                avg_time = sum(r["time"] for r in successful) / len(successful)
                avg_size = sum(r["size"] for r in successful) / len(successful)

                status = "✅ GOOD" if avg_time < 30 else "⚠️ SLOW" if avg_time < 120 else "❌ TOO SLOW"

                print(f"{config_file:<30} {avg_time:>6.2f}s     {avg_size/1024:>6.1f} KB    {status}")
            else:
                print(f"{config_file:<30} {'FAILED':<12} {'N/A':<12} ❌")

    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    print("For production mining:")
    print("  - Use text_mv_fast.yaml if it achieves <25s")
    print("  - Use text_mv_ultra_fast.yaml if you need <15s")
    print("  - Monitor CLIP scores on dashboard to ensure >0.6")
    print("  - Adjust iters parameter to balance speed vs quality")
    print("\nCooldown windows:")
    print("  - 300s cooldown: Need <30s generation")
    print("  - 120s cooldown (UID49): Need <15s generation")
    print("="*60)


if __name__ == "__main__":
    main()
