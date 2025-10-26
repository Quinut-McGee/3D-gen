"""
Mesh to Gaussian Splat Converter
CRITICAL COMPONENT for competitive mining on Bittensor subnet 17

Converts triangle meshes (from InstantMesh) to Gaussian Splat PLY format
that validators expect. This is the competitive edge that enables sub-30s generation.

Target: 3-5 seconds for conversion
"""

import numpy as np
import trimesh
from typing import Tuple, Optional
from io import BytesIO
import struct
import logging
import torch

logger = logging.getLogger(__name__)


class MeshToGaussianConverter:
    """
    Converts triangle mesh to Gaussian Splat PLY format.

    This converter samples points from the mesh surface and computes
    Gaussian parameters (position, scale, rotation, opacity, color) that
    produce a plausible 3D representation.

    Performance target: 3-5 seconds per conversion
    """

    def __init__(self, num_gaussians: int = 8000, base_scale: float = 0.005):
        """
        Initialize converter with configuration.

        Args:
            num_gaussians: Number of Gaussian primitives to generate (7000-8000 recommended)
            base_scale: Base scale for Gaussian primitives (affects "splat" size)
        """
        self.num_gaussians = num_gaussians
        self.base_scale = base_scale
        self.last_model = None  # Cache the last created GaussianModel for rendering

        logger.info(f"MeshToGaussianConverter initialized: {num_gaussians} gaussians, base_scale={base_scale}")

    def convert(self, mesh: trimesh.Trimesh, rgba_image=None, num_gaussians: int = None) -> BytesIO:
        """
        Convert triangle mesh to Gaussian Splat PLY format.

        TARGET PERFORMANCE: 3-5 seconds

        Args:
            mesh: Input trimesh.Trimesh object
            rgba_image: PIL Image (RGBA) for 2D color sampling - REQUIRED!
            num_gaussians: Override default number of gaussians

        Returns:
            BytesIO containing PLY file data

        Process:
        1. Sample points uniformly from mesh surface (Poisson disk sampling)
        2. Compute normals at sampled points
        3. Sample colors from 2D RGBA image via projection (CRITICAL FOR QUALITY!)
        4. Compute scales based on local surface density
        5. Compute rotations from normals (align with surface)
        6. Set reasonable opacities
        7. Convert to spherical harmonics (DC component only)
        8. Export to binary PLY format
        """
        import time
        start_time = time.time()

        if rgba_image is None:
            raise ValueError("rgba_image is required for 2D color sampling! Cannot produce quality results without it.")

        if num_gaussians is None:
            num_gaussians = self.num_gaussians

        logger.info(f"Converting mesh to {num_gaussians} Gaussian primitives...")

        # Step 1: Sample points from surface
        points, face_indices = self._sample_surface_points(mesh, num_gaussians)

        # Step 2: Compute normals at sampled points
        normals = self._compute_normals(mesh, points, face_indices)

        # Step 3: Sample colors from 2D RGBA image (NEW METHOD - critical for quality!)
        logger.info("  Sampling colors from 2D RGBA image...")
        colors = self._sample_colors_from_2d_image(points, rgba_image, mesh)
        logger.info(f"  Sampled {len(colors)} colors from image")

        # Step 4: Compute scales
        scales = self._compute_scales(mesh, points, face_indices, self.base_scale)

        # Step 5: Compute rotations from normals
        rotations = self._compute_rotations(normals)

        # Step 6: Compute opacities
        opacities = self._compute_opacities(mesh, points, face_indices)

        # Step 7: Convert colors to spherical harmonics (DC component)
        f_dc = self._rgb_to_sh(colors)

        # Step 7.5: Create and cache GaussianModel for rendering validation
        try:
            from DreamGaussianLib.GaussianSplattingModel import GaussianModel

            gs_model = GaussianModel(sh_degree=1)

            # CRITICAL: Apply inverse transforms for Gaussian Splatting
            # Opacity must be in inverse sigmoid space
            from scipy.special import logit
            opacities_clipped = np.clip(opacities, 0.001, 0.999)  # Avoid infinities at 0/1
            opacities_inverse_sigmoid = logit(opacities_clipped)

            # Scales must be in log space
            scales_log = np.log(scales + 1e-8)  # Avoid log(0)

            # Populate with computed data (with proper transforms)
            gs_model._xyz = torch.from_numpy(points).float().cuda()
            gs_model._features_dc = torch.from_numpy(f_dc).float().cuda().unsqueeze(1)  # Shape: [N, 1, 3]
            gs_model._opacity = torch.from_numpy(opacities_inverse_sigmoid).float().cuda().unsqueeze(1)  # Shape: [N, 1], inverse sigmoid
            gs_model._scaling = torch.from_numpy(scales_log).float().cuda()  # Shape: [N, 3], log space
            gs_model._rotation = torch.from_numpy(rotations).float().cuda()  # Shape: [N, 4], quaternions

            # DEBUG: Log Gaussian data stats
            logger.debug(f"Gaussian data stats:")
            logger.debug(f"  positions: shape={gs_model._xyz.shape}, range=[{gs_model._xyz.min():.3f}, {gs_model._xyz.max():.3f}]")
            logger.debug(f"  scales: shape={gs_model._scaling.shape}, range=[{gs_model._scaling.min():.3f}, {gs_model._scaling.max():.3f}]")
            logger.debug(f"  opacity: shape={gs_model._opacity.shape}, range=[{gs_model._opacity.min():.3f}, {gs_model._opacity.max():.3f}]")
            logger.debug(f"  rotation: shape={gs_model._rotation.shape}, range=[{gs_model._rotation.min():.3f}, {gs_model._rotation.max():.3f}]")
            logger.debug(f"  features_dc: shape={gs_model._features_dc.shape}, range=[{gs_model._features_dc.min():.3f}, {gs_model._features_dc.max():.3f}]")

            # Cache for rendering
            self.last_model = gs_model
            logger.debug(f"Cached GaussianModel with {len(points)} gaussians for rendering")

        except Exception as e:
            logger.warning(f"Failed to create GaussianModel (rendering unavailable): {e}")
            self.last_model = None

        # Step 8: Export to PLY
        ply_data = self._export_to_ply(points, normals, f_dc, opacities, scales, rotations)

        elapsed = time.time() - start_time
        logger.info(f"Mesh to Gaussian conversion completed in {elapsed:.2f}s")

        return ply_data

    def get_last_model(self):
        """
        Get the last created GaussianModel for rendering validation.

        Returns:
            GaussianModel or None if no model has been created yet
        """
        return self.last_model

    def _sample_surface_points(self, mesh: trimesh.Trimesh, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points uniformly from mesh surface using area-weighted sampling.

        Args:
            mesh: Input mesh
            num_points: Number of points to sample

        Returns:
            points: [N, 3] sampled point positions
            face_indices: [N] indices of faces each point was sampled from
        """
        # Use trimesh's built-in sampling (area-weighted)
        points, face_indices = trimesh.sample.sample_surface(mesh, num_points)

        logger.debug(f"Sampled {len(points)} points from {len(mesh.faces)} faces")

        return points, face_indices

    def _compute_normals(self, mesh: trimesh.Trimesh, points: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
        """
        Compute normals at sampled points using face normals.

        Args:
            mesh: Input mesh
            points: Sampled points
            face_indices: Face indices for each point

        Returns:
            normals: [N, 3] normal vectors (normalized)
        """
        # Get face normals for sampled faces
        face_normals = mesh.face_normals[face_indices]

        # Normalize (should already be normalized, but ensure it)
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        normals = face_normals / norms

        return normals

    def _extract_colors(self, mesh: trimesh.Trimesh, points: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
        """
        Extract RGB colors at sampled points.

        Args:
            mesh: Input mesh
            points: Sampled points
            face_indices: Face indices for each point

        Returns:
            colors: [N, 3] RGB colors in range [0, 1]
        """
        if mesh.visual is not None and hasattr(mesh.visual, 'vertex_colors'):
            # Mesh has vertex colors - interpolate to sample points
            # Get vertices of sampled faces
            face_vertices = mesh.faces[face_indices]  # [N, 3] indices

            # Get vertex colors for each face
            v0_colors = mesh.visual.vertex_colors[face_vertices[:, 0]]
            v1_colors = mesh.visual.vertex_colors[face_vertices[:, 1]]
            v2_colors = mesh.visual.vertex_colors[face_vertices[:, 2]]

            # Simple average (could use barycentric, but average is faster)
            colors = (v0_colors + v1_colors + v2_colors) / 3.0

            # Convert to [0, 1] if in [0, 255]
            if colors.max() > 1.0:
                colors = colors / 255.0

        else:
            # No vertex colors - use default neutral gray
            colors = np.ones((len(points), 3)) * 0.5

        return colors[:, :3]  # Ensure RGB only

    def _compute_scales(
        self,
        mesh: trimesh.Trimesh,
        points: np.ndarray,
        face_indices: np.ndarray,
        base_scale: float
    ) -> np.ndarray:
        """
        Compute scale parameters for Gaussians based on local surface density.

        Args:
            mesh: Input mesh
            points: Sampled points
            face_indices: Face indices for each point
            base_scale: Base scale factor

        Returns:
            scales: [N, 3] scale parameters (one per axis)
        """
        # Get face areas for local density estimation
        face_areas = mesh.area_faces[face_indices]

        # Estimate local scale from face area
        # Larger faces → larger scales (sparser sampling needs bigger splats)
        local_scales = np.sqrt(face_areas) * base_scale

        # Use isotropic scaling (same scale in all directions)
        scales = np.stack([local_scales, local_scales, local_scales], axis=1)

        # Clamp scales to reasonable range
        scales = np.clip(scales, 0.001, 0.1)

        return scales

    def _compute_rotations(self, normals: np.ndarray) -> np.ndarray:
        """
        Compute rotation quaternions that align Gaussians with surface normals.

        Args:
            normals: [N, 3] surface normals

        Returns:
            quaternions: [N, 4] rotation quaternions [w, x, y, z]
        """
        # We want to rotate the Gaussian from default orientation (aligned with Z-axis)
        # to align with the surface normal

        # Default direction (Z-axis)
        default_dir = np.array([0, 0, 1])

        # Compute rotation axis (cross product)
        rotation_axes = np.cross(np.tile(default_dir, (len(normals), 1)), normals)
        rotation_axis_norms = np.linalg.norm(rotation_axes, axis=1, keepdims=True)

        # Compute rotation angle (dot product)
        cos_angles = np.dot(normals, default_dir)
        angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))

        # Normalize rotation axes
        # Handle case where normal is already aligned (rotation_axis_norm ≈ 0)
        small_norm_mask = rotation_axis_norms[:, 0] < 1e-6
        rotation_axes[~small_norm_mask] /= rotation_axis_norms[~small_norm_mask]

        # Convert axis-angle to quaternion
        half_angles = angles / 2.0
        sin_half_angles = np.sin(half_angles)
        cos_half_angles = np.cos(half_angles)

        quaternions = np.zeros((len(normals), 4))
        quaternions[:, 0] = cos_half_angles  # w
        quaternions[:, 1:4] = rotation_axes * sin_half_angles[:, np.newaxis]  # x, y, z

        # For aligned normals, use identity quaternion
        quaternions[small_norm_mask] = [1, 0, 0, 0]

        return quaternions

    def _compute_opacities(self, mesh: trimesh.Trimesh, points: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
        """
        Compute opacity values for Gaussians.

        Args:
            mesh: Input mesh
            points: Sampled points
            face_indices: Face indices for each point

        Returns:
            opacities: [N] opacity values in range [0, 1]
        """
        # For surface sampling, we want high opacity (we're confident about surface)
        # Higher opacity = more solid/visible Gaussians in renders
        opacities = np.ones(len(points)) * 0.99

        return opacities

    def _sample_colors_from_2d_image(self, points: np.ndarray, rgba_image, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Sample colors from 2D RGBA image by projecting 3D points.

        This is the CRITICAL METHOD for getting correct colors from FLUX output.
        Instead of extracting non-existent vertex colors from InstantMesh,
        we project 3D Gaussian positions back to 2D and sample from the FLUX image.

        Args:
            points: [N, 3] array of 3D positions
            rgba_image: PIL Image (RGBA) from FLUX → background removal
            mesh: trimesh.Trimesh for bounds calculation

        Returns:
            colors: [N, 3] array of RGB colors in [0, 1] range
        """
        from PIL import Image

        logger.debug(f"  Sampling colors from 2D image: {rgba_image.size}")

        # Convert image to numpy array
        img_array = np.array(rgba_image.convert('RGB')).astype(np.float32) / 255.0
        h, w = img_array.shape[:2]

        # Get mesh bounding box for normalization
        bbox = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        center = (bbox[0] + bbox[1]) / 2.0
        scale = (bbox[1] - bbox[0]).max()

        # Normalize points to [-1, 1] range
        normalized_points = (points - center) / (scale / 2.0)

        # Orthographic projection: use X and Y, ignore Z
        # Map [-1, 1] → [0, width-1] and [0, height-1]
        u = ((normalized_points[:, 0] + 1.0) * 0.5 * (w - 1)).astype(np.int32)
        v = ((1.0 - (normalized_points[:, 1] + 1.0) * 0.5) * (h - 1)).astype(np.int32)  # Flip Y axis

        # Clamp to valid image bounds
        u = np.clip(u, 0, w - 1)
        v = np.clip(v, 0, h - 1)

        # Sample colors from image
        colors = img_array[v, u]  # Shape: [N, 3], RGB in [0, 1]

        # Handle transparency: points on background get default color
        if rgba_image.mode == 'RGBA':
            alpha_array = np.array(rgba_image)[:, :, 3].astype(np.float32) / 255.0
            alpha_sampled = alpha_array[v, u]

            # Where alpha < 0.1 (transparent background), use neutral gray
            background_mask = alpha_sampled < 0.1
            colors[background_mask] = 0.5  # Neutral gray for background

            logger.debug(f"  Found {background_mask.sum()} background points")

        logger.debug(f"  Color range: [{colors.min():.3f}, {colors.max():.3f}]")

        return colors

    def _rgb_to_sh(self, colors: np.ndarray) -> np.ndarray:
        """
        Convert RGB colors to spherical harmonics DC component.

        For Gaussian Splatting, colors are represented as spherical harmonics.
        DC component (l=0, m=0) is a constant offset.

        Args:
            colors: [N, 3] RGB colors in range [0, 1]

        Returns:
            f_dc: [N, 3] SH DC components
        """
        # DC component is RGB * constant factor
        # SH_C0 = 0.28209479177387814 (1 / (2 * sqrt(pi)))
        SH_C0 = 0.28209479177387814

        # Convert RGB to SH DC
        # We need to reverse the transformation: RGB = f_dc * SH_C0
        # Therefore: f_dc = RGB / SH_C0
        f_dc = colors / SH_C0

        return f_dc

    def _export_to_ply(
        self,
        positions: np.ndarray,
        normals: np.ndarray,
        f_dc: np.ndarray,
        opacities: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray,
    ) -> BytesIO:
        """
        Export Gaussian parameters to binary PLY format.

        PLY format matches DreamGaussian output for validator compatibility.

        Args:
            positions: [N, 3] positions
            normals: [N, 3] normals
            f_dc: [N, 3] spherical harmonics DC component
            opacities: [N] opacities
            scales: [N, 3] scales
            rotations: [N, 4] quaternions [w, x, y, z]

        Returns:
            BytesIO containing PLY data
        """
        num_points = len(positions)

        # Create PLY header
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float f_rest_0
property float f_rest_1
property float f_rest_2
property float f_rest_3
property float f_rest_4
property float f_rest_5
property float f_rest_6
property float f_rest_7
property float f_rest_8
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
""".encode('utf-8')

        # Prepare binary data
        # f_rest: We only use DC component, so f_rest is all zeros (9 components for l=1,2,3)
        f_rest = np.zeros((num_points, 9), dtype=np.float32)

        # Pack all data into structured array
        vertex_data = np.zeros(
            num_points,
            dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
                ('f_rest_0', 'f4'), ('f_rest_1', 'f4'), ('f_rest_2', 'f4'),
                ('f_rest_3', 'f4'), ('f_rest_4', 'f4'), ('f_rest_5', 'f4'),
                ('f_rest_6', 'f4'), ('f_rest_7', 'f4'), ('f_rest_8', 'f4'),
                ('opacity', 'f4'),
                ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
                ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
            ]
        )

        # Fill data
        vertex_data['x'] = positions[:, 0]
        vertex_data['y'] = positions[:, 1]
        vertex_data['z'] = positions[:, 2]
        vertex_data['nx'] = normals[:, 0]
        vertex_data['ny'] = normals[:, 1]
        vertex_data['nz'] = normals[:, 2]
        vertex_data['f_dc_0'] = f_dc[:, 0]
        vertex_data['f_dc_1'] = f_dc[:, 1]
        vertex_data['f_dc_2'] = f_dc[:, 2]
        for i in range(9):
            vertex_data[f'f_rest_{i}'] = f_rest[:, i]
        vertex_data['opacity'] = opacities
        vertex_data['scale_0'] = scales[:, 0]
        vertex_data['scale_1'] = scales[:, 1]
        vertex_data['scale_2'] = scales[:, 2]
        vertex_data['rot_0'] = rotations[:, 0]  # w
        vertex_data['rot_1'] = rotations[:, 1]  # x
        vertex_data['rot_2'] = rotations[:, 2]  # y
        vertex_data['rot_3'] = rotations[:, 3]  # z

        # Write to BytesIO
        ply_buffer = BytesIO()
        ply_buffer.write(header)
        ply_buffer.write(vertex_data.tobytes())
        ply_buffer.seek(0)

        logger.debug(f"Exported PLY with {num_points} Gaussians")

        return ply_buffer


# Quick test function
if __name__ == "__main__":
    # Test with a simple mesh
    print("Testing MeshToGaussianConverter...")

    # Create a simple test mesh (sphere)
    test_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)

    # Add some vertex colors
    test_mesh.visual.vertex_colors = np.random.randint(0, 255, (len(test_mesh.vertices), 4))

    # Initialize converter
    converter = MeshToGaussianConverter(num_gaussians=1000)

    # Convert
    ply_data = converter.convert(test_mesh)

    # Check size
    ply_size = len(ply_data.getvalue())
    print(f"✅ Test successful!")
    print(f"Generated PLY size: {ply_size / 1024:.1f} KB")
    print(f"Expected ~{1000 * 27 * 4 / 1024:.1f} KB for 1000 gaussians (27 floats each)")
