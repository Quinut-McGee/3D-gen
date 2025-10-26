#!/usr/bin/env python3
"""
Standalone DreamGaussian test to verify it works in isolation
"""
import os
import sys

# CRITICAL: Add ninja to PATH before importing anything
conda_bin = '/home/kobe/miniconda3/envs/three-gen-mining/bin'
if conda_bin not in os.environ.get('PATH', ''):
    os.environ['PATH'] = f"{conda_bin}:{os.environ.get('PATH', '')}"

from PIL import Image
from DreamGaussianLib import ModelsPreLoader
from DreamGaussianLib.GaussianProcessor import GaussianProcessor
from omegaconf import OmegaConf
import tempfile
import os

print("=" * 60)
print("STANDALONE DREAMGAUSSIAN TEST")
print("=" * 60)

# Use a known-good RGBA image from debug output (red sports car)
print("\n[1/5] Loading RGBA image...")
rgba_image = Image.open("/tmp/debug_2_rembg_1761337808.png")
print(f"✅ Loaded image: {rgba_image.size} {rgba_image.mode}")

# Save to temp file
print("\n[2/5] Preparing temp files...")
with tempfile.NamedTemporaryFile(suffix='_rgba.png', delete=False) as tmp:
    rgba_image.save(tmp.name)
    tmp_path = tmp.name
    print(f"✅ Saved to: {tmp_path}")

# Create caption file
caption_path = tmp_path.replace('_rgba.png', '_caption.txt')
with open(caption_path, 'w') as f:
    f.write("red sports car")
print(f"✅ Caption: {caption_path}")

try:
    # Load DreamGaussian with ORIGINAL config
    print("\n[3/5] Loading DreamGaussian models...")
    config = OmegaConf.load("configs/text_mv_fast.yaml")
    config.input = tmp_path
    config.prompt = "red sports car"
    print(f"Config iterations: {config.iters}")

    # Load models
    device = "cuda"
    models = ModelsPreLoader.preload_model(config, device)
    print("✅ Models loaded")

    # Train with config iterations
    print(f"\n[4/5] Training DreamGaussian ({config.iters} iterations)...")
    processor = GaussianProcessor(config, "red sports car")
    processor.train(models, config.iters)
    print("✅ Training complete")

    # Save output
    print("\n[5/5] Saving Gaussian Splat...")
    output_path = "/tmp/test_standalone_dreamgaussian.ply"
    with open(output_path, "wb") as f:
        processor.get_gs_model().save_ply(f)

    # Check file size
    file_size = os.path.getsize(output_path)
    print(f"✅ Saved: {output_path}")
    print(f"   Size: {file_size / 1024:.1f} KB")

    print("\n" + "=" * 60)
    print("✅ STANDALONE TEST COMPLETE")
    print("=" * 60)
    print(f"\nOutput PLY: {output_path}")
    print("Next: Render this PLY to check quality")

finally:
    # Clean up temp files
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    if os.path.exists(caption_path):
        os.remove(caption_path)
