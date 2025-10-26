#!/usr/bin/env python3
"""
Test CLIP validator with the actual FLUX-generated images to verify scoring.

This will help us determine if:
1. CLIP validator is broken
2. FLUX images are actually low quality
3. Prompts are the issue
"""
import sys
sys.path.insert(0, '/home/kobe/404-gen/v1/3D-gen/generation')

from validators.clip_validator import CLIPValidator
from PIL import Image

print("=" * 80)
print("CLIP VALIDATOR DIAGNOSTIC TEST")
print("=" * 80)

# Initialize CLIP
print("\n[1/3] Loading CLIP validator...")
validator = CLIPValidator(device="cuda", threshold=0.6)
validator.to_gpu()
print("✅ CLIP loaded")

# Test 1: Simple color test
print("\n[2/3] Test 1: Simple red square (baseline)")
red_square = Image.new('RGB', (512, 512), color='red')
_, score1 = validator.validate_image(red_square, "red color")
print(f"   Red square vs 'red color': {score1:.3f}")

# Test 2: FLUX-generated image with ORIGINAL simple prompt
print("\n[3/3] Test 2: FLUX-generated red sports car")
flux_image = Image.open("/tmp/debug_1_flux_1761495363.png")

# Test with SIMPLE prompt (what user typed)
_, score_simple = validator.validate_image(flux_image, "red sports car")
print(f"   FLUX image vs 'red sports car' (simple): {score_simple:.3f}")

# Test with ENHANCED prompt (what was actually used)
enhanced_prompt = "a photorealistic red sports car, professional product photography, studio lighting setup, pure white background, centered composition, sharp focus, highly detailed, 8k resolution, award-winning photography"
_, score_enhanced = validator.validate_image(flux_image, enhanced_prompt)
print(f"   FLUX image vs enhanced prompt: {score_enhanced:.3f}")

# Test with even simpler prompts
_, score_car = validator.validate_image(flux_image, "car")
print(f"   FLUX image vs 'car': {score_car:.3f}")

_, score_vehicle = validator.validate_image(flux_image, "red vehicle")
print(f"   FLUX image vs 'red vehicle': {score_vehicle:.3f}")

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)

if score_simple < 0.3:
    print("❌ PROBLEM: Even simple prompts score LOW (<0.3)")
    print("   This suggests CLIP validator might have an issue")
    print("   OR FLUX images don't match prompts well")
elif score_simple < 0.5:
    print("⚠️  MODERATE: Simple prompt scores 0.3-0.5")
    print("   FLUX is generating recognizable content but quality is borderline")
elif score_simple >= 0.5:
    print("✅ GOOD: Simple prompt scores >0.5")
    print("   CLIP validator is working, prompt engineering may be the issue")

if score_enhanced < score_simple:
    print(f"\n⚠️  Enhanced prompt HURTS score by {(score_simple - score_enhanced)*100:.1f}%")
    print("   Recommendation: Use SIMPLE prompts!")
else:
    print(f"\n✅ Enhanced prompt HELPS score by {(score_enhanced - score_simple)*100:.1f}%")

print("\nExpected CLIP scores for good images: 0.5-0.7")
print(f"Your scores: {score_simple:.3f} (simple), {score_enhanced:.3f} (enhanced)")
