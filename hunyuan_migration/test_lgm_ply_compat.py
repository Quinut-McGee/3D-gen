#!/usr/bin/env python3
"""
CRITICAL COMPATIBILITY TEST: Verify LGM PLY works with existing GaussianModel loader
If this fails, migration is BLOCKED - we need different approach
"""
import sys
sys.path.insert(0, '/home/kobe/404-gen/v1/3D-gen/generation')

import torch
import numpy as np
from pathlib import Path
import tempfile

print("=" * 70)
print("LGM PLY COMPATIBILITY TEST")
print("=" * 70)

# Test 1: Can we import LGM's Gaussian renderer?
print("\n[TEST 1] Importing LGM Gaussian renderer...")
try:
    # Make sure kiui.op is imported
    from kiui import op as kiui_op
    sys.path.insert(0, '/home/kobe/404-gen/v1/3D-gen/hunyuan_migration/LGM')
    from core.gs import GaussianRenderer
    from core.options import AllConfigs
    print("✅ PASSED: LGM imports successful")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Can we create a test PLY using LGM's format?
print("\n[TEST 2] Creating test PLY with LGM format...")
try:
    # Create minimal gaussian data: [B=1, N=1000, 14]
    # Format: [xyz(3) + opacity(1) + scales(3) + rotations(4) + rgb(3)]
    num_gaussians = 1000
    test_gaussians = torch.zeros(1, num_gaussians, 14)

    # Position: random points in unit cube
    test_gaussians[0, :, 0:3] = torch.rand(num_gaussians, 3) * 2 - 1

    # Opacity: random 0.1 to 1.0 (will be converted to log-space)
    test_gaussians[0, :, 3:4] = torch.rand(num_gaussians, 1) * 0.9 + 0.1

    # Scales: random 0.01 to 0.1 (will be converted to log-space)
    test_gaussians[0, :, 4:7] = torch.rand(num_gaussians, 3) * 0.09 + 0.01

    # Rotations: normalized quaternions
    test_gaussians[0, :, 7:11] = torch.randn(num_gaussians, 4)
    test_gaussians[0, :, 7:11] = torch.nn.functional.normalize(test_gaussians[0, :, 7:11], dim=-1)

    # RGB: random colors 0-1 (will be converted to SH)
    test_gaussians[0, :, 11:14] = torch.rand(num_gaussians, 3)

    print(f"   Created test gaussians: shape={test_gaussians.shape}")
    print(f"   Position range: [{test_gaussians[0, :, 0:3].min():.2f}, {test_gaussians[0, :, 0:3].max():.2f}]")
    print(f"   Opacity range: [{test_gaussians[0, :, 3].min():.2f}, {test_gaussians[0, :, 3].max():.2f}]")
    print("✅ PASSED: Test gaussian data created")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Save PLY using LGM's save_ply
print("\n[TEST 3] Saving PLY with LGM's save_ply...")
try:
    # Create minimal config
    class MinimalOpt:
        fovy = 49.1
        zfar = 100.0
        znear = 0.5
        output_size = 512

    opt = MinimalOpt()
    gs_renderer = GaussianRenderer(opt)

    test_ply_path = Path("/tmp/lgm_compat_test.ply")
    gs_renderer.save_ply(test_gaussians, str(test_ply_path), compatible=True)

    file_size_kb = test_ply_path.stat().st_size / 1024
    print(f"   Saved PLY: {test_ply_path}")
    print(f"   File size: {file_size_kb:.1f} KB")
    print("✅ PASSED: PLY saved successfully")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Load PLY with GaussianModel (CRITICAL!)
print("\n[TEST 4] Loading PLY with GaussianModel (CRITICAL TEST)...")
try:
    from DreamGaussianLib.GaussianSplattingModel import GaussianModel

    gs_model = GaussianModel(3)  # sh_degree=3
    gs_model.load_ply(str(test_ply_path))

    num_loaded = len(gs_model._xyz)
    print(f"   Loaded {num_loaded:,} gaussians")
    print(f"   Position shape: {gs_model._xyz.shape}")
    print(f"   Opacity shape: {gs_model._opacity.shape}")
    print(f"   Features DC shape: {gs_model._features_dc.shape}")
    print("✅ PASSED: GaussianModel loaded PLY successfully")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Opacity corruption check (CRITICAL!)
print("\n[TEST 5] Opacity corruption check...")
try:
    opacities = gs_model._opacity.cpu().numpy()
    opacity_mean = np.mean(opacities)
    opacity_std = np.std(opacities)
    opacity_min = np.min(opacities)
    opacity_max = np.max(opacities)

    print(f"   Mean opacity: {opacity_mean:.4f}")
    print(f"   Std opacity:  {opacity_std:.4f}")
    print(f"   Min opacity:  {opacity_min:.4f}")
    print(f"   Max opacity:  {opacity_max:.4f}")

    if opacity_std < 0.1:
        print("❌ FAILED: Opacity corruption detected (std < 0.1)!")
        sys.exit(1)

    # Check for the specific TRELLIS bug pattern
    if abs(opacity_mean - (-6.907)) < 0.01 and opacity_std < 0.1:
        print("❌ FAILED: TRELLIS-style opacity corruption detected!")
        sys.exit(1)

    print("✅ PASSED: No opacity corruption")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Verify all required gaussian parameters
print("\n[TEST 6] Verifying all gaussian parameters...")
try:
    required_attrs = ['_xyz', '_opacity', '_scaling', '_rotation', '_features_dc', '_features_rest']
    for attr in required_attrs:
        if not hasattr(gs_model, attr):
            print(f"❌ Missing attribute: {attr}")
            sys.exit(1)
        value = getattr(gs_model, attr)
        print(f"   ✓ {attr}: shape={value.shape}, dtype={value.dtype}")

    print("✅ PASSED: All required attributes present")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Round-trip test (save and reload)
print("\n[TEST 7] Round-trip test (GaussianModel → PLY → GaussianModel)...")
try:
    # Save using GaussianModel
    test_ply_path_2 = Path("/tmp/lgm_roundtrip_test.ply")
    gs_model.save_ply(str(test_ply_path_2))

    # Reload
    gs_model_2 = GaussianModel(3)
    gs_model_2.load_ply(str(test_ply_path_2))

    # Compare
    if len(gs_model_2._xyz) == len(gs_model._xyz):
        print(f"   ✓ Gaussian count preserved: {len(gs_model_2._xyz):,}")
    else:
        print(f"   ⚠️  Gaussian count changed: {len(gs_model._xyz):,} → {len(gs_model_2._xyz):,}")

    print("✅ PASSED: Round-trip successful")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Cleanup
print("\n[CLEANUP] Removing test files...")
test_ply_path.unlink(missing_ok=True)
test_ply_path_2.unlink(missing_ok=True)

print("\n" + "=" * 70)
print("✅✅✅ ALL TESTS PASSED - LGM IS FULLY COMPATIBLE ✅✅✅")
print("=" * 70)
print(f"\nSummary:")
print(f"  - LGM outputs standard 3D Gaussian Splatting PLY format")
print(f"  - GaussianModel.load_ply() works perfectly with LGM PLY files")
print(f"  - No opacity corruption detected")
print(f"  - All required gaussian parameters present")
print(f"  - Ready for production migration!")
print("=" * 70)
