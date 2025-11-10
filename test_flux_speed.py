#!/usr/bin/env python3
"""Test FLUX/SDXL-Turbo speed in isolation"""

import sys
sys.path.insert(0, '/home/kobe/404-gen/v1/3D-gen/generation')

import time
import torch
from models.sdxl_turbo_generator import SDXLTurboGenerator

print("=" * 70)
print("FLUX/SDXL-Turbo Speed Test")
print("=" * 70)

# Initialize generator
print("\n1. Initializing SDXL-Turbo on cuda:1...")
t0 = time.time()
gen = SDXLTurboGenerator(device='cuda:1')
init_time = time.time() - t0
print(f"   Initialization time: {init_time:.2f}s")

# Test prompts (simple to complex)
test_prompts = [
    "test cube",
    "red sports car",
    "ceramic dollhouse pastel hues victorian architecture",
    "a slender, lustrous pearl pendant necklace with a delicate chain"
]

results = []

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{i}. Testing: '{prompt}'")
    print(f"   Steps: 4, Size: 512x512")

    t0 = time.time()
    img = gen.generate(
        prompt=prompt,
        num_inference_steps=4,
        height=512,
        width=512,
        seed=42
    )
    gen_time = time.time() - t0

    print(f"   ✅ Generated in {gen_time:.2f}s")
    results.append((prompt, gen_time))

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for prompt, gen_time in results:
    print(f"{gen_time:6.2f}s - '{prompt}'")

avg_time = sum(t for _, t in results) / len(results)
print(f"\nAverage generation time: {avg_time:.2f}s")

# Diagnosis
print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)
if avg_time < 5.0:
    print("✅ EXCELLENT: FLUX is performing as expected (< 5s)")
elif avg_time < 10.0:
    print("⚠️  ACCEPTABLE: FLUX is slightly slow (5-10s), but usable")
elif avg_time < 20.0:
    print("⚠️  SLOW: FLUX is slower than expected (10-20s)")
    print("   Possible causes: GPU contention, thermal throttling, or driver issues")
else:
    print("❌ CRITICAL: FLUX is extremely slow (>20s)")
    print("   This is 4-5x slower than expected!")
    print("   Possible causes:")
    print("   - Wrong inference steps being used")
    print("   - GPU memory pressure causing swapping")
    print("   - Incorrect device assignment")

print(f"\nExpected: 1-4s per generation (SDXL-Turbo with 4 steps)")
print(f"Actual: {avg_time:.2f}s")
print("=" * 70)
