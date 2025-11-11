"""
Clean PLY Writer - Bypasses TRELLIS save_ply() Corruption

Root Cause: TRELLIS save_ply() applies inverse_sigmoid() and torch.log()
transformations that introduce numerical instability, corrupting opacity values
by 1.2-11.2 points even after our normalization fixes them.

This writer:
1. Extracts Gaussian parameters directly from gaussian_output
2. Applies proper activations to get actual values
3. Writes PLY without inverse operations that cause corruption
4. Verifies written values match expected values
"""

import torch
import numpy as np
from plyfile import PlyData, PlyElement
from loguru import logger


def save_clean_ply(gaussian_output, path: str):
    """
    Save Gaussian splat to PLY file without TRELLIS corruption.

    Args:
        gaussian_output: TRELLIS Gaussian object with normalized _opacity
        path: Output PLY file path
    """
    # Extract raw parameters (these are in "hidden" space with biases)
    _xyz = gaussian_output._xyz
    _opacity = gaussian_output._opacity
    _scaling = gaussian_output._scaling
    _rotation = gaussian_output._rotation
    _features_dc = gaussian_output._features_dc
    _features_rest = gaussian_output._features_rest

    # Get biases from gaussian object
    opacity_bias = gaussian_output.opacity_bias
    scale_bias = gaussian_output.scale_bias
    rots_bias = gaussian_output.rots_bias
    aabb = gaussian_output.aabb
    mininum_kernel_size = gaussian_output.mininum_kernel_size

    # Apply activations to get ACTUAL values (what will be rendered)
    # Position: denormalize from AABB
    xyz = (_xyz * aabb[None, 3:] + aabb[None, :3]).detach().cpu().numpy()

    # Opacity: apply sigmoid to get [0,1] range for PLY storage
    # CRITICAL FIX: Store opacity in SIGMOID SPACE [0, 1], not logit space!
    # GaussianModel.load_ply() expects [0, 1] and will apply inverse_sigmoid internally
    # Previous bug: We were applying inverse_sigmoid here, causing double transformation
    actual_opacity = torch.sigmoid(_opacity + opacity_bias)

    # Log opacity for verification
    opacity_before = actual_opacity.detach().cpu().numpy()
    logger.info(f"🔬 CLEAN PLY WRITER - Opacity (sigmoid space [0,1]):")
    logger.info(f"   mean: {opacity_before.mean():.4f}, std: {opacity_before.std():.4f}")
    logger.info(f"   min: {opacity_before.min():.4f}, max: {opacity_before.max():.4f}")

    # CRITICAL FIX: Write opacity in SIGMOID SPACE [0, 1]
    # DO NOT apply inverse_sigmoid here! GaussianModel.load_ply() will handle that.
    # Previous bug: Applying inverse_sigmoid here + load_ply() applying it again = double transformation
    opacities_ply = actual_opacity.detach().cpu().numpy()

    # Scaling: apply activation to get actual scale, then log for PLY
    scales = torch.exp(_scaling + scale_bias)
    scales = torch.square(scales) + mininum_kernel_size ** 2
    scales = torch.sqrt(scales)
    scale_ply = torch.log(scales).detach().cpu().numpy()

    # Rotation: normalize and add bias
    rotation = torch.nn.functional.normalize(_rotation + rots_bias[None, :])
    rotation_ply = rotation.detach().cpu().numpy()

    # Features: DC component (RGB color)
    f_dc = _features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

    # Normals (zeros - not used in Gaussian splatting)
    normals = np.zeros_like(xyz)

    # Apply transform (TRELLIS default: reorient coordinate system)
    transform = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    xyz = np.matmul(xyz, transform.T)

    # Transform rotations
    import utils3d
    rotation_matrices = utils3d.numpy.quaternion_to_matrix(rotation_ply)
    rotation_matrices = np.matmul(transform, rotation_matrices)
    rotation_ply = utils3d.numpy.matrix_to_quaternion(rotation_matrices)

    # Construct PLY attributes
    attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(f_dc.shape[1]):
        attributes.append(f'f_dc_{i}')
    attributes.append('opacity')
    for i in range(scale_ply.shape[1]):
        attributes.append(f'scale_{i}')
    for i in range(rotation_ply.shape[1]):
        attributes.append(f'rot_{i}')

    # Create structured array
    dtype_full = [(attr, 'f4') for attr in attributes]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)

    # Concatenate all attributes
    all_attributes = np.concatenate((xyz, normals, f_dc, opacities_ply, scale_ply, rotation_ply), axis=1)
    elements[:] = list(map(tuple, all_attributes))

    # Write PLY file
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

    logger.info(f"✅ Clean PLY written to {path}")

    # VERIFICATION: Re-read and check for corruption
    try:
        plydata = PlyData.read(path)
        saved_opacities = plydata['vertex']['opacity']

        # CRITICAL FIX: Opacity is now stored in SIGMOID SPACE [0, 1]
        # No need to apply sigmoid transformation - just read directly
        saved_actual = saved_opacities

        logger.info(f"🔬 CLEAN PLY WRITER - Opacity after write (re-read from disk):")
        logger.info(f"   mean: {saved_actual.mean():.4f}, std: {saved_actual.std():.4f}")
        logger.info(f"   min: {saved_actual.min():.4f}, max: {saved_actual.max():.4f}")

        # Calculate corruption delta
        corruption_delta = abs(saved_actual.mean() - opacity_before.mean())
        logger.info(f"   corruption_delta: {corruption_delta:.4f}")

        if corruption_delta > 0.01:
            logger.warning(f"   ⚠️ CORRUPTION DETECTED: opacity changed by {corruption_delta:.4f} during write!")
        else:
            logger.info(f"   ✅ No corruption detected (delta < 0.01)")

    except Exception as e:
        logger.warning(f"   Could not verify PLY write: {e}")

    return path
