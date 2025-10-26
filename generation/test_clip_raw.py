#!/usr/bin/env python3
"""
Test raw CLIP scoring to verify expected values.
Compare against known benchmarks.
"""
import torch
import clip
from PIL import Image

print("Testing raw CLIP scores...")

# Load CLIP
device = "cuda"
model, preprocess = clip.load("ViT-L/14", device=device)

# Test 1: Red square
print("\n[Test 1] Red square vs 'red color'")
red_square = Image.new('RGB', (512, 512), color='red')
image = preprocess(red_square).unsqueeze(0).to(device)
text = clip.tokenize(["red color"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # Normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Cosine similarity
    similarity = (image_features @ text_features.T).item()

    print(f"Score: {similarity:.3f}")
    print(f"Logit score (x100): {similarity * 100:.1f}")

# Test 2: FLUX image
print("\n[Test 2] FLUX car image vs different prompts")
car_image = Image.open("/tmp/debug_1_flux_1761495363.png")
image = preprocess(car_image).unsqueeze(0).to(device)

prompts = [
    "red sports car",
    "a red sports car",
    "car",
    "vehicle",
    "red color",
    "a photo of a red sports car",
]

with torch.no_grad():
    image_features = model.encode_image(image)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    for prompt in prompts:
        text = clip.tokenize([prompt]).to(device)
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).item()
        print(f"   '{prompt}': {similarity:.3f}")

print("\n" + "=" * 60)
print("INTERPRETATION:")
print("=" * 60)
print("CLIP scores are typically in range [0.0, 1.0]")
print("- 0.20-0.25: Very weak match")
print("- 0.25-0.30: Weak match")
print("- 0.30-0.40: Moderate match")
print("- 0.40-0.50: Good match")
print("- 0.50+: Strong match")
print("\nIf ALL scores are <0.3, CLIP model might be misconfigured")
