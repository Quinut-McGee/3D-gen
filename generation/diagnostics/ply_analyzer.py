"""
PLY Quality Analyzer - Deep diagnostics for gaussian splat quality
Analyzes spatial distribution, not just count
"""
import struct
import numpy as np
from typing import Dict, Optional
from loguru import logger


def analyze_gaussian_quality(ply_bytes: bytes) -> Dict[str, float]:
    """
    Analyze PLY quality beyond just gaussian count.

    Returns metrics that might correlate with validator Score=0.0:
    - Spatial distribution (clumped vs evenly distributed)
    - Opacity statistics (transparent vs solid)
    - Scale statistics (too small = invisible, too large = blobs)
    - Bounding box volume (collapsed model detection)
    """
    try:
        # Parse PLY header
        header_end = ply_bytes.find(b"end_header\n")
        if header_end == -1:
            logger.error("Invalid PLY: no end_header marker")
            return {}

        header = ply_bytes[:header_end].decode('utf-8', errors='ignore')
        data_start = header_end + len(b"end_header\n")

        # Extract vertex count
        num_vertices = 0
        for line in header.split('\n'):
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
                break

        if num_vertices == 0:
            logger.error("No vertices found in PLY")
            return {}

        # Parse binary data (assuming binary PLY format)
        # Standard gaussian splat PLY format:
        # x, y, z (position) - 3 floats
        # nx, ny, nz (normal) - 3 floats (often unused)
        # f_dc_0, f_dc_1, f_dc_2 (color) - 3 floats
        # opacity - 1 float
        # scale_0, scale_1, scale_2 - 3 floats
        # rot_0, rot_1, rot_2, rot_3 - 4 floats (quaternion)
        # Total: 17 floats per vertex = 68 bytes per vertex

        bytes_per_vertex = 68  # Standard format
        expected_bytes = num_vertices * bytes_per_vertex
        actual_bytes = len(ply_bytes) - data_start

        if actual_bytes < expected_bytes * 0.9:  # Allow 10% tolerance
            logger.warning(f"PLY data size mismatch: expected ~{expected_bytes}, got {actual_bytes}")
            # Try to infer bytes per vertex
            bytes_per_vertex = actual_bytes // num_vertices

        # Extract positions, scales, opacities
        positions = []
        scales = []
        opacities = []

        for i in range(min(num_vertices, actual_bytes // bytes_per_vertex)):
            offset = data_start + i * bytes_per_vertex

            try:
                # Position (x, y, z)
                x, y, z = struct.unpack_from('fff', ply_bytes, offset)
                positions.append([x, y, z])

                # Skip normals (3 floats) and color (3 floats)
                # Opacity is at offset +36 bytes (9 floats * 4 bytes)
                opacity = struct.unpack_from('f', ply_bytes, offset + 36)[0]
                opacities.append(opacity)

                # Scales at offset +40 bytes (10 floats * 4 bytes)
                scale_0, scale_1, scale_2 = struct.unpack_from('fff', ply_bytes, offset + 40)
                scales.append([scale_0, scale_1, scale_2])

            except struct.error:
                # Reached end of data
                break

        if len(positions) < 100:
            logger.error(f"Too few valid vertices parsed: {len(positions)}")
            return {}

        # Convert to numpy for analysis
        positions = np.array(positions)
        scales = np.array(scales)
        opacities = np.array(opacities)

        # METRIC 1: Spatial Distribution Variance
        # High variance = evenly distributed, Low variance = clumped
        position_std = np.std(positions, axis=0)  # Std dev per axis
        spatial_variance = float(np.mean(position_std))

        # METRIC 2: Bounding Box Volume
        # Too small = collapsed model
        bbox_min = np.min(positions, axis=0)
        bbox_max = np.max(positions, axis=0)
        bbox_size = bbox_max - bbox_min
        bbox_volume = float(np.prod(bbox_size))

        # METRIC 3: Opacity Statistics
        # Too transparent = invisible model
        avg_opacity = float(np.mean(opacities))
        min_opacity = float(np.min(opacities))
        opacity_std = float(np.std(opacities))

        # METRIC 4: Scale Statistics
        # Too small = invisible, too large = blobs
        scale_magnitudes = np.linalg.norm(scales, axis=1)
        avg_scale = float(np.mean(scale_magnitudes))
        min_scale = float(np.min(scale_magnitudes))
        max_scale = float(np.max(scale_magnitudes))
        scale_std = float(np.std(scale_magnitudes))

        # METRIC 5: Density Uniformity
        # Check if gaussians are clustered or evenly distributed
        # Divide space into grid and count gaussians per cell
        grid_size = 10
        hist, _ = np.histogramdd(positions, bins=[grid_size, grid_size, grid_size])
        density_variance = float(np.std(hist))
        density_max = float(np.max(hist))

        metrics = {
            'num_gaussians': len(positions),
            'spatial_variance': spatial_variance,
            'bbox_volume': bbox_volume,
            'bbox_size_x': float(bbox_size[0]),
            'bbox_size_y': float(bbox_size[1]),
            'bbox_size_z': float(bbox_size[2]),
            'avg_opacity': avg_opacity,
            'min_opacity': min_opacity,
            'opacity_std': opacity_std,
            'avg_scale': avg_scale,
            'min_scale': min_scale,
            'max_scale': max_scale,
            'scale_std': scale_std,
            'density_variance': density_variance,
            'density_max': density_max,
        }

        logger.debug(f"PLY Quality Metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"PLY analysis failed: {e}", exc_info=True)
        return {}


def diagnose_ply_issues(metrics: Dict[str, float]) -> list:
    """
    Identify potential quality issues based on metrics.
    Returns list of diagnostic messages.
    """
    issues = []

    if metrics.get('bbox_volume', 0) < 0.01:
        issues.append("⚠️ COLLAPSED MODEL: Bounding box volume too small")

    if metrics.get('avg_opacity', 1.0) < 0.3:
        issues.append("⚠️ TOO TRANSPARENT: Average opacity below 0.3")

    if metrics.get('spatial_variance', 0) < 0.1:
        issues.append("⚠️ CLUMPED GAUSSIANS: Low spatial variance, gaussians not distributed")

    if metrics.get('avg_scale', 0) < 0.001:
        issues.append("⚠️ TOO SMALL: Average gaussian scale below visibility threshold")

    # DISABLED: False positive - compares L2 norm of log-space scales to exp-space threshold
    # The diagnostic calculated L2 norm of log-space 3D vectors (e.g., sqrt((-7)^2*3) = 12.12)
    # and compared to 10.0, triggering false positives on normal-sized gaussians
    # if metrics.get('max_scale', 0) > 10.0:
    #     issues.append("⚠️ TOO LARGE: Some gaussians are blob-sized")

    if metrics.get('density_variance', 0) > metrics.get('density_max', 1) * 0.5:
        issues.append("⚠️ UNEVEN DENSITY: Gaussians heavily clustered in some regions")

    return issues
