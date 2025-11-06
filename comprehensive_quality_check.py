#!/usr/bin/env python3
"""
Comprehensive Quality Test for Generated PLY Files
Checks for opacity issues, background removal problems, spatial degeneration,
and other quality concerns that could lead to validator rejection (score 0)
"""

import struct
import numpy as np
from pathlib import Path
import sys

def read_ply_file(filepath):
    """Read PLY file and extract gaussian properties"""
    # Check if file is empty
    if filepath.stat().st_size == 0:
        raise ValueError("Empty PLY file")

    with open(filepath, 'rb') as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        # Parse vertex count
        vertex_count = 0
        for line in header_lines:
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
                break

        # Read binary data
        # Format: x, y, z, nx, ny, nz, f_dc_0-2, opacity, scale_0-2, rot_0-3
        bytes_per_vertex = 17 * 4  # 17 floats per gaussian
        data = f.read()

        # Sample every Nth gaussian for speed (analyze 10000 samples)
        sample_step = max(1, vertex_count // 10000)

        gaussians = []
        for i in range(0, vertex_count, sample_step):
            offset = i * bytes_per_vertex
            if offset + bytes_per_vertex > len(data):
                break
            vertex_data = struct.unpack('f' * 17, data[offset:offset + bytes_per_vertex])

            gaussians.append({
                'position': vertex_data[0:3],   # x, y, z
                'normal': vertex_data[3:6],     # nx, ny, nz
                'color_dc': vertex_data[6:9],   # f_dc_0, f_dc_1, f_dc_2 (base color)
                'opacity': vertex_data[9],      # opacity value
                'scale': vertex_data[10:13],    # scale_0, scale_1, scale_2
                'rotation': vertex_data[13:17]  # rot_0, rot_1, rot_2, rot_3 (quaternion)
            })

        return gaussians, vertex_count

def analyze_opacity(gaussians):
    """Check for opacity corruption"""
    opacities = np.array([g['opacity'] for g in gaussians])

    issues = []

    # Check 1: Opacity variation (critical for visibility)
    opacity_std = np.std(opacities)
    opacity_mean = np.mean(opacities)

    if opacity_std < 1.0:
        issues.append(f"❌ CRITICAL: Opacity flattening detected (std={opacity_std:.3f}, should be >3.0)")
    elif opacity_std < 3.0:
        issues.append(f"⚠️  WARNING: Low opacity variation (std={opacity_std:.3f}, prefer >3.0)")
    else:
        issues.append(f"✅ Healthy opacity variation (std={opacity_std:.3f})")

    # Check 2: Opacity range
    opacity_min = np.min(opacities)
    opacity_max = np.max(opacities)

    if opacity_min == opacity_max:
        issues.append(f"❌ CRITICAL: All opacities identical (frozen at {opacity_min:.3f})")
    elif opacity_max - opacity_min < 2.0:
        issues.append(f"⚠️  WARNING: Narrow opacity range ({opacity_min:.3f} to {opacity_max:.3f})")
    else:
        issues.append(f"✅ Good opacity range ({opacity_min:.3f} to {opacity_max:.3f})")

    # Check 3: Average opacity (should be visible, not transparent)
    if opacity_mean < -5.0:
        issues.append(f"❌ CRITICAL: Model mostly invisible (avg opacity={opacity_mean:.3f})")
    elif opacity_mean < 0.0:
        issues.append(f"⚠️  WARNING: Low average opacity ({opacity_mean:.3f})")
    else:
        issues.append(f"✅ Good average opacity ({opacity_mean:.3f})")

    return issues, opacity_std, opacity_mean

def analyze_spatial_quality(gaussians):
    """Check for spatial degeneration (collapsed models)"""
    positions = np.array([g['position'] for g in gaussians])

    issues = []

    # Check 1: Spatial extent (bounding box)
    bbox_min = np.min(positions, axis=0)
    bbox_max = np.max(positions, axis=0)
    bbox_size = bbox_max - bbox_min

    volume = np.prod(bbox_size)

    if volume < 0.1:
        issues.append(f"❌ CRITICAL: Model collapsed to tiny volume ({volume:.6f})")
    elif volume < 1.0:
        issues.append(f"⚠️  WARNING: Small spatial extent (volume={volume:.3f})")
    else:
        issues.append(f"✅ Good spatial extent (volume={volume:.3f})")

    # Check 2: Spatial distribution variance
    spatial_std = np.std(positions, axis=0)
    avg_spatial_std = np.mean(spatial_std)

    if avg_spatial_std < 0.1:
        issues.append(f"❌ CRITICAL: Gaussians clustered at single point (std={avg_spatial_std:.6f})")
    elif avg_spatial_std < 0.5:
        issues.append(f"⚠️  WARNING: Low spatial distribution (std={avg_spatial_std:.3f})")
    else:
        issues.append(f"✅ Good spatial distribution (std={avg_spatial_std:.3f})")

    return issues, volume

def analyze_background_removal(gaussians):
    """Check background removal quality via color variation"""
    colors = np.array([g['color_dc'] for g in gaussians])

    issues = []

    # Check color variation (good background removal = diverse colors, not uniform)
    color_std = np.std(colors, axis=0)
    avg_color_std = np.mean(color_std)

    if avg_color_std < 0.01:
        issues.append(f"❌ CRITICAL: No color variation (all same color, avg_std={avg_color_std:.6f})")
    elif avg_color_std < 0.1:
        issues.append(f"⚠️  WARNING: Low color variation (avg_std={avg_color_std:.3f})")
    else:
        issues.append(f"✅ Good color variation (avg_std={avg_color_std:.3f})")

    return issues

def analyze_scale_degeneration(gaussians):
    """Check for abnormally tiny or huge gaussians"""
    scales = np.array([g['scale'] for g in gaussians])

    issues = []

    # Check scale statistics
    scale_mean = np.mean(scales)
    scale_std = np.std(scales)
    scale_min = np.min(scales)
    scale_max = np.max(scales)

    # Abnormally tiny scales (invisible points)
    if scale_mean < 0.001:
        issues.append(f"❌ CRITICAL: Abnormally tiny gaussians (avg scale={scale_mean:.6f})")
    elif scale_mean < 0.01:
        issues.append(f"⚠️  WARNING: Very small gaussians (avg scale={scale_mean:.6f})")
    else:
        issues.append(f"✅ Reasonable gaussian sizes (avg scale={scale_mean:.6f})")

    # Scale variation
    if scale_std < 0.0001:
        issues.append(f"⚠️  WARNING: All gaussians same size (no depth variation)")
    else:
        issues.append(f"✅ Good scale variation (std={scale_std:.6f})")

    return issues

def analyze_file(filepath):
    """Comprehensive analysis of single PLY file"""
    print(f"\n{'='*80}")
    print(f"File: {filepath.name}")
    print(f"{'='*80}")

    try:
        gaussians, count = read_ply_file(filepath)
        print(f"Gaussian count: {count:,}")

        # Density check
        density_status = "✅" if count >= 300000 else ("⚠️" if count >= 150000 else "❌")
        print(f"{density_status} Density: {count:,} gaussians", end="")
        if count < 150000:
            print(" (TOO SPARSE - high rejection risk)")
        elif count < 300000:
            print(" (acceptable but prefer 300K+)")
        else:
            print(" (excellent)")

        print()

        # Run all quality checks
        all_issues = []

        print("1. OPACITY ANALYSIS:")
        opacity_issues, opacity_std, opacity_mean = analyze_opacity(gaussians)
        for issue in opacity_issues:
            print(f"   {issue}")
            all_issues.append(issue)
        print()

        print("2. SPATIAL QUALITY:")
        spatial_issues, volume = analyze_spatial_quality(gaussians)
        for issue in spatial_issues:
            print(f"   {issue}")
            all_issues.append(issue)
        print()

        print("3. BACKGROUND REMOVAL:")
        bg_issues = analyze_background_removal(gaussians)
        for issue in bg_issues:
            print(f"   {issue}")
            all_issues.append(issue)
        print()

        print("4. SCALE ANALYSIS:")
        scale_issues = analyze_scale_degeneration(gaussians)
        for issue in scale_issues:
            print(f"   {issue}")
            all_issues.append(issue)
        print()

        # Overall verdict for this file
        critical_count = len([i for i in all_issues if '❌' in i])
        warning_count = len([i for i in all_issues if '⚠️' in i])

        if critical_count > 0:
            verdict = "❌ REJECTED - Critical issues detected"
        elif warning_count > 2:
            verdict = "⚠️  QUESTIONABLE - Multiple warnings"
        else:
            verdict = "✅ PRODUCTION READY"

        print(f"FILE VERDICT: {verdict}")
        print(f"  Critical issues: {critical_count}")
        print(f"  Warnings: {warning_count}")

        return {
            'filename': filepath.name,
            'count': count,
            'opacity_std': opacity_std,
            'opacity_mean': opacity_mean,
            'volume': volume,
            'critical_issues': critical_count,
            'warnings': warning_count,
            'verdict': verdict
        }

    except Exception as e:
        print(f"❌ ERROR reading file: {e}")
        return {
            'filename': filepath.name,
            'error': str(e),
            'critical_issues': 1,
            'warnings': 0,
            'verdict': '❌ READ ERROR'
        }

def main():
    print("=" * 80)
    print("COMPREHENSIVE QUALITY ANALYSIS - PHASE 1 TEST GENERATIONS")
    print("=" * 80)
    print()
    print("Analyzing generated PLY files for:")
    print("  • Opacity corruption (most critical)")
    print("  • Spatial degeneration (collapsed models)")
    print("  • Background removal quality")
    print("  • Gaussian density")
    print("  • Scale degeneration")
    print()

    # Find all test PLY files
    test_files = sorted(Path('/tmp').glob('test_gen_*.ply'))

    if not test_files:
        print("❌ No test files found in /tmp/test_gen_*.ply")
        print("Run ./test_phase1_fixes.sh first to generate test files")
        return 1

    print(f"Found {len(test_files)} test files")
    print()

    # Analyze each file
    results = []
    for filepath in test_files[:10]:  # First 10 files
        result = analyze_file(filepath)
        results.append(result)

    # Summary statistics
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}\n")

    valid_results = [r for r in results if 'error' not in r]

    if valid_results:
        avg_count = int(np.mean([r['count'] for r in valid_results]))
        avg_opacity_std = np.mean([r['opacity_std'] for r in valid_results])

        print(f"Files analyzed: {len(valid_results)}")
        print(f"Average gaussian count: {avg_count:,}")
        print(f"Average opacity std: {avg_opacity_std:.3f}")
        print()

        # Count verdicts
        production_ready = len([r for r in valid_results if '✅ PRODUCTION READY' in r['verdict']])
        questionable = len([r for r in valid_results if '⚠️' in r['verdict']])
        rejected = len([r for r in valid_results if '❌' in r['verdict']])

        print(f"PRODUCTION READY: {production_ready}/{len(valid_results)} ({production_ready/len(valid_results)*100:.1f}%)")
        print(f"QUESTIONABLE:     {questionable}/{len(valid_results)} ({questionable/len(valid_results)*100:.1f}%)")
        print(f"REJECTED:         {rejected}/{len(valid_results)} ({rejected/len(valid_results)*100:.1f}%)")
        print()

        # Overall recommendation
        if rejected > 0:
            print("⚠️  RECOMMENDATION: Some files have critical issues - review before deployment")
        elif questionable > len(valid_results) // 2:
            print("⚠️  RECOMMENDATION: Many warnings detected - consider further optimization")
        else:
            print("✅ RECOMMENDATION: Quality looks good - ready for mainnet deployment")

        # Expected performance
        print()
        print("EXPECTED VALIDATOR PERFORMANCE:")
        if avg_opacity_std >= 3.0 and avg_count >= 300000:
            print("  • Opacity corruption: ✅ ELIMINATED (std >= 3.0)")
            print("  • Gaussian density: ✅ STRONG (avg >= 300K)")
            print("  • Predicted success rate: 50-70% (up from 21%)")
            print("  • Score 0 risk: VERY LOW")
        elif avg_opacity_std >= 3.0:
            print("  • Opacity corruption: ✅ ELIMINATED (std >= 3.0)")
            print("  • Gaussian density: ⚠️  ACCEPTABLE but could be higher")
            print("  • Predicted success rate: 40-60%")
            print("  • Score 0 risk: LOW-MEDIUM")
        else:
            print("  • Opacity corruption: ❌ STILL PRESENT")
            print("  • Predicted success rate: <30%")
            print("  • Score 0 risk: HIGH")

    print()
    print("=" * 80)

    return 0

if __name__ == '__main__':
    sys.exit(main())
