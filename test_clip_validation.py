"""
Test Phase 3: CLIP Validation Test
Tests multiple diverse prompts to verify CLIP scores and performance.
"""

import requests
import time
import json
from tabulate import tabulate

# Test prompts - diverse objects to test generalization
TEST_PROMPTS = [
    "a red sports car, sleek design, photorealistic",
    "a wooden chair, modern design, minimalist",
    "a blue ceramic teapot, glossy finish",
    "a modern desk lamp, metallic, contemporary",
    "a leather couch, brown, comfortable",
]

API_URL = "http://localhost:8080/generate/"

def test_generation(prompt):
    """Run a single generation and return results"""
    print(f"\n{'='*80}")
    print(f"Testing: {prompt}")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        response = requests.post(
            API_URL,
            data={"prompt": prompt},
            timeout=120
        )

        total_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()

            return {
                "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                "success": True,
                "total_time": f"{total_time:.2f}s",
                "clip_score": result.get("clip_score", "N/A"),
                "passed_validation": result.get("passed_validation", "N/A"),
                "gaussians": result.get("num_gaussians", "N/A"),
                "file_size_mb": result.get("file_size_mb", "N/A"),
                "flux_time": f"{result.get('flux_time', 0):.2f}s",
                "trellis_time": f"{result.get('trellis_time', 0):.2f}s",
            }
        else:
            return {
                "prompt": prompt[:50],
                "success": False,
                "error": f"HTTP {response.status_code}",
                "total_time": f"{total_time:.2f}s"
            }

    except Exception as e:
        total_time = time.time() - start_time
        return {
            "prompt": prompt[:50],
            "success": False,
            "error": str(e),
            "total_time": f"{total_time:.2f}s"
        }


def main():
    print("\n" + "="*80)
    print("TEST PHASE 3: CLIP VALIDATION TEST")
    print("="*80)
    print(f"Testing {len(TEST_PROMPTS)} diverse prompts")
    print(f"Target: CLIP score > 0.65, Time < 30s, No OOM")
    print("="*80)

    results = []

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}]", end=" ")
        result = test_generation(prompt)
        results.append(result)

        # Print immediate result
        if result["success"]:
            print(f"✅ CLIP: {result['clip_score']:.3f}, Time: {result['total_time']}, Passed: {result['passed_validation']}")
        else:
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")

        # Brief pause between tests
        time.sleep(2)

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if results:
        # Filter successful results
        success_results = [r for r in results if r.get("success")]

        if success_results:
            # Display table
            table_data = []
            for r in success_results:
                table_data.append([
                    r["prompt"],
                    r["clip_score"],
                    r["passed_validation"],
                    r["total_time"],
                    r["gaussians"],
                    f"{r['file_size_mb']:.1f} MB" if isinstance(r['file_size_mb'], (int, float)) else r['file_size_mb']
                ])

            headers = ["Prompt", "CLIP Score", "Passed", "Time", "Gaussians", "File Size"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))

            # Statistics
            clip_scores = [r["clip_score"] for r in success_results if isinstance(r["clip_score"], (int, float))]
            if clip_scores:
                avg_clip = sum(clip_scores) / len(clip_scores)
                min_clip = min(clip_scores)
                max_clip = max(clip_scores)

                print(f"\n{'='*80}")
                print(f"CLIP STATISTICS:")
                print(f"  Average: {avg_clip:.3f}")
                print(f"  Min:     {min_clip:.3f}")
                print(f"  Max:     {max_clip:.3f}")
                print(f"  Target:  > 0.650")
                print(f"  Result:  {'✅ PASS' if avg_clip > 0.65 else '❌ FAIL'}")

                passed_count = sum(1 for r in success_results if r.get("passed_validation"))
                print(f"\n{'='*80}")
                print(f"VALIDATION STATISTICS:")
                print(f"  Passed: {passed_count}/{len(success_results)} ({passed_count/len(success_results)*100:.0f}%)")
                print(f"  Failed: {len(success_results)-passed_count}/{len(success_results)}")

                print(f"\n{'='*80}")
                print("FINAL VERDICT:")
                if avg_clip > 0.65 and passed_count >= len(success_results) * 0.8:
                    print("✅ TEST PHASE 3 PASSED!")
                    print("   - CLIP scores are competitive")
                    print("   - Validation passing at high rate")
                    print("   - Ready for mainnet deployment!")
                else:
                    print("⚠️  TEST PHASE 3 NEEDS REVIEW")
                    if avg_clip <= 0.65:
                        print(f"   - CLIP scores below target (avg: {avg_clip:.3f})")
                    if passed_count < len(success_results) * 0.8:
                        print(f"   - Validation pass rate too low ({passed_count/len(success_results)*100:.0f}%)")
                print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
