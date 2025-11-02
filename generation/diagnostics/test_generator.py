#!/usr/bin/env python3
"""
Test Generation Harness - Generate and evaluate test cases

Run test generations with different prompts to understand:
1. What types of prompts work well
2. What quality patterns emerge
3. How to optimize your pipeline

Usage:
    python test_generator.py --prompts "red car,blue chair,green apple" --count 5
"""

import asyncio
import argparse
import httpx
import time
import base64
from pathlib import Path
from loguru import logger


# Test prompt sets for different categories
TEST_PROMPTS = {
    'simple_objects': [
        'red sports car',
        'blue wooden chair',
        'green apple',
        'silver fork',
        'yellow banana',
    ],
    'complex_objects': [
        'ornate golden chandelier with crystals',
        'vintage leather armchair with brass studs',
        'mechanical steampunk pocket watch',
        'intricate Celtic knot sculpture',
        'detailed marble statue of a lion',
    ],
    'jewelry': [
        'diamond engagement ring',
        'pearl necklace',
        'gold bracelet',
        'silver earrings',
        'sapphire pendant',
    ],
    'tools': [
        'hammer with wooden handle',
        'screwdriver with rubber grip',
        'adjustable wrench',
        'pliers with red handles',
        'tape measure',
    ],
}


async def generate_test_case(prompt: str, endpoint="http://localhost:10010/generate/"):
    """Generate a single test case"""
    logger.info(f"Testing prompt: '{prompt}'")

    try:
        start_time = time.time()

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Text-to-3D request
            response = await client.post(
                endpoint,
                json={
                    "type": "text-to-3d",
                    "prompt": prompt,
                }
            )

            if response.status_code != 200:
                logger.error(f"Generation failed: {response.status_code}")
                return None

            ply_bytes = response.content
            generation_time = time.time() - start_time

            # Get metrics from headers
            quality_score = response.headers.get('X-CLIP-Score', 'N/A')

            result = {
                'prompt': prompt,
                'success': True,
                'generation_time': generation_time,
                'ply_size_kb': len(ply_bytes) / 1024,
                'clip_score': quality_score,
            }

            logger.info(f"✅ Generated in {generation_time:.1f}s, size={result['ply_size_kb']:.1f}KB")
            return result

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return {
            'prompt': prompt,
            'success': False,
            'error': str(e)
        }


async def run_test_suite(prompts: list, iterations: int = 1):
    """Run a suite of test generations"""
    logger.info(f"Running test suite: {len(prompts)} prompts × {iterations} iterations")

    results = []
    for iteration in range(iterations):
        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {iteration + 1}/{iterations}")
        logger.info(f"{'='*60}\n")

        for prompt in prompts:
            result = await generate_test_case(prompt)
            if result:
                results.append(result)

            # Small delay between generations
            await asyncio.sleep(2)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST SUITE COMPLETE")
    logger.info(f"{'='*60}\n")

    successful = sum(1 for r in results if r.get('success'))
    logger.info(f"Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")

    if successful > 0:
        avg_time = sum(r['generation_time'] for r in results if r.get('success')) / successful
        avg_size = sum(r['ply_size_kb'] for r in results if r.get('success')) / successful
        logger.info(f"Average generation time: {avg_time:.1f}s")
        logger.info(f"Average PLY size: {avg_size:.1f}KB")

    logger.info(f"\n💡 Now run: python diagnostics/self_evaluation.py")
    logger.info(f"   to see quality analysis of these test generations")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test Generation Harness")
    parser.add_argument(
        '--category',
        choices=list(TEST_PROMPTS.keys()) + ['all'],
        default='simple_objects',
        help='Category of test prompts to use'
    )
    parser.add_argument(
        '--prompts',
        type=str,
        help='Custom comma-separated list of prompts'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=1,
        help='Number of iterations per prompt'
    )
    parser.add_argument(
        '--endpoint',
        type=str,
        default='http://localhost:10010/generate/',
        help='Generation endpoint'
    )

    args = parser.parse_args()

    # Determine prompts to use
    if args.prompts:
        prompts = [p.strip() for p in args.prompts.split(',')]
    elif args.category == 'all':
        prompts = []
        for category_prompts in TEST_PROMPTS.values():
            prompts.extend(category_prompts)
    else:
        prompts = TEST_PROMPTS[args.category]

    logger.info(f"Selected {len(prompts)} prompts")

    # Run async test suite
    asyncio.run(run_test_suite(prompts, args.count))


if __name__ == "__main__":
    main()
