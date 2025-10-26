#!/usr/bin/env python3
"""
Test rendering a single red Gaussian to verify renderer works
"""
import os
import sys

# Add ninja to PATH
conda_bin = '/home/kobe/miniconda3/envs/three-gen-mining/bin'
if conda_bin not in os.environ.get('PATH', ''):
    os.environ['PATH'] = f"{conda_bin}:{os.environ.get('PATH', '')}"

import torch
import numpy as np
from scipy.special import logit
from DreamGaussianLib.GaussianSplattingModel import GaussianModel

print("=" * 60)
print("MINIMAL GAUSSIAN RENDER TEST")
print("=" * 60)

# Create a minimal Gaussian Splat manually
print("\n[1/3] Creating single red Gaussian at origin...")
model = GaussianModel(sh_degree=0)

# 1 Gaussian at origin
model._xyz = torch.tensor([[0.0, 0.0, 0.0]], device='cuda', dtype=torch.float32)
print(f"✅ Position: {model._xyz.shape} = [[0, 0, 0]]")

# Red color (SH encoding)
# RGB (1, 0, 0) → SH DC component
SH_C0 = 0.28209479177387814
red_sh = torch.tensor([[(1.0 - 0.5) / SH_C0, (0.0 - 0.5) / SH_C0, (0.0 - 0.5) / SH_C0]],
                       device='cuda', dtype=torch.float32)
model._features_dc = red_sh.unsqueeze(1)  # [1, 1, 3]
print(f"✅ Color (SH DC): {model._features_dc.shape}")

# Rest of SH (zeros for sh_degree=0, none needed)
model._features_rest = torch.zeros((1, 0, 3), device='cuda', dtype=torch.float32)

# Full opacity (inverse sigmoid of 0.99)
opacity_value = logit(0.99)
model._opacity = torch.tensor([[opacity_value]], device='cuda', dtype=torch.float32)
print(f"✅ Opacity: {model._opacity.shape} = [[{opacity_value:.2f}]] (inverse sigmoid of 0.99)")

# Medium scale (log space) - make it visible
scale_value = np.log(0.3)  # Larger scale so it's visible
model._scaling = torch.tensor([[scale_value, scale_value, scale_value]],
                              device='cuda', dtype=torch.float32)
print(f"✅ Scale: {model._scaling.shape} = [[{scale_value:.2f}]] (log of 0.3)")

# Identity rotation (w=1, x=y=z=0)
model._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device='cuda', dtype=torch.float32)
print(f"✅ Rotation: {model._rotation.shape} = [[1, 0, 0, 0]] (identity quaternion)")

# Set active_sh_degree
model.active_sh_degree = 0
model.max_sh_degree = 0

print(f"\n📊 Model summary:")
print(f"   Total Gaussians: 1")
print(f"   Position: origin (0,0,0)")
print(f"   Color: RED")
print(f"   Size: 0.3 units radius")
print(f"   Opacity: 99%")

# Now render this single red Gaussian
print("\n[2/3] Rendering single Gaussian...")
try:
    from rendering.quick_render import render_gaussian_model_to_images

    views = render_gaussian_model_to_images(
        model=model,
        num_views=1,
        resolution=512,
        device='cuda'
    )

    if views and len(views) > 0:
        output_path = '/tmp/test_single_red_gaussian.png'
        views[0].save(output_path)
        print(f"✅ Saved render to: {output_path}")
        print(f"   Image size: {views[0].size}")
        print(f"   Image mode: {views[0].mode}")

        # Check if image is actually red or just gray
        import numpy as np
        img_array = np.array(views[0])
        mean_color = img_array.mean(axis=(0, 1))
        print(f"   Mean RGB: [{mean_color[0]:.1f}, {mean_color[1]:.1f}, {mean_color[2]:.1f}]")

        if mean_color[0] > mean_color[1] and mean_color[0] > mean_color[2]:
            print("   ✅ Image is reddish - renderer works!")
        else:
            print("   ❌ Image is NOT red - renderer might be broken")

        print("\n[3/3] Result:")
        print("=" * 60)
        if mean_color[0] > 100:
            print("✅ SUCCESS: Renderer correctly displays a red Gaussian!")
            print("   → DreamGaussian is likely creating valid Gaussians")
            print("   → Problem might be with DreamGaussian parameters or config")
        else:
            print("❌ FAILURE: Renderer shows gray/dark image")
            print("   → Rendering function is fundamentally broken")
            print("   → Need to fix renderer before testing 3D methods")
        print("=" * 60)
    else:
        print("❌ Rendering returned no views")

except Exception as e:
    print(f"❌ Rendering failed with error: {e}")
    import traceback
    traceback.print_exc()
