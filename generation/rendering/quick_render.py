"""
Quick PLY rendering for validation.
Uses existing GaussianRenderer for fast multi-view rendering.
"""

import os
import io
import torch
import numpy as np
from PIL import Image
from typing import List, Optional
from loguru import logger

# Ensure ninja is in PATH before importing CUDA extensions
conda_bin = '/home/kobe/miniconda3/envs/three-gen-mining/bin'
if conda_bin not in os.environ.get('PATH', ''):
    os.environ['PATH'] = f"{conda_bin}:{os.environ.get('PATH', '')}"

# Import existing Gaussian rendering infrastructure
from DreamGaussianLib.rendering.gs_camera import OrbitCamera
from DreamGaussianLib.rendering.gs_renderer import GaussianRenderer
from DreamGaussianLib.GaussianSplattingModel import GaussianModel


def render_ply_to_images(
    ply_bytes: bytes,
    num_views: int = 4,
    resolution: int = 512,
    device: str = "cuda"
) -> Optional[List[Image.Image]]:
    """
    Render a PLY file to multiple views for CLIP validation.

    Args:
        ply_bytes: Raw PLY file content
        num_views: Number of views to render (default 4 for MVDream)
        resolution: Image resolution
        device: CUDA device

    Returns:
        List of PIL Images, or None on error
    """
    try:
        # Load PLY into Gaussian model
        model = GaussianModel(sh_degree=0)

        # Parse PLY from bytes
        ply_buffer = io.BytesIO(ply_bytes)
        model.load_ply(ply_buffer)

        # Create renderer
        renderer = GaussianRenderer()

        # Generate views at different angles
        images = []
        angles = np.linspace(0, 360, num_views, endpoint=False)

        for i, angle in enumerate(angles):
            logger.debug(f"  Rendering view {i+1}/{num_views} (angle={angle:.1f}°)...")

            # Create camera
            camera = OrbitCamera(resolution, resolution, fov_y=49.1)
            camera.compute_transform_orbit(
                elevation=0,
                azimuth=angle,
                radius=3.0
            )

            # Get Gaussian data
            means3D = model.get_xyz
            rotations = model.get_rotation
            scales = model.get_scaling
            opacity = model.get_opacity.squeeze(1)
            features = model.get_features_dc.transpose(1, 2).flatten(start_dim=1).contiguous()

            # Convert to RGB
            SH_C0 = 0.28209479177387814
            rgbs = (0.5 + SH_C0 * features)

            gs_data = [means3D, rotations, scales, opacity, rgbs]

            # Render
            with torch.no_grad():
                image, _, _, _ = renderer.render(
                    camera.world_to_camera_transform.unsqueeze(0),
                    camera.intrinsics.unsqueeze(0),
                    (camera.image_width, camera.image_height),
                    camera.z_near,
                    camera.z_far,
                    gs_data
                )

            # Convert to PIL Image
            image_np = image[0].cpu().numpy()
            image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)

            images.append(pil_image)

            # CRITICAL: Clean up GPU memory after each view to prevent OOM
            del image, image_np, gs_data, means3D, rotations, scales, opacity, rgbs, features
            del camera  # Free camera tensors too

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.debug(f"  View {i+1}/{num_views} rendered successfully, GPU cache cleared")

        return images

    except Exception as e:
        logger.error(f"Failed to render PLY: {e}")
        return None


def render_gaussian_model_to_images(
    model,  # GaussianModel type
    num_views: int = 4,
    resolution: int = 512,
    device: str = "cuda"
) -> Optional[List[Image.Image]]:
    """
    Render a GaussianModel to multiple views (no PLY loading needed).

    This is more efficient than render_ply_to_images() because it uses
    the existing model in memory instead of reloading from PLY bytes.
    """
    try:
        # Create renderer
        renderer = GaussianRenderer()

        # Generate views at different angles
        images = []
        angles = np.linspace(0, 360, num_views, endpoint=False)

        for i, angle in enumerate(angles):
            logger.debug(f"  Rendering view {i+1}/{num_views} (angle={angle:.1f}°)...")

            # Create camera
            camera = OrbitCamera(resolution, resolution, fov_y=49.1)
            camera.compute_transform_orbit(
                elevation=0,
                azimuth=angle,
                radius=3.0
            )

            # Get Gaussian data from existing model
            means3D = model.get_xyz
            rotations = model.get_rotation
            scales = model.get_scaling
            opacity = model.get_opacity.squeeze(1)
            features = model.get_features_dc.transpose(1, 2).flatten(start_dim=1).contiguous()

            # Convert to RGB
            SH_C0 = 0.28209479177387814
            rgbs = (0.5 + SH_C0 * features)

            gs_data = [means3D, rotations, scales, opacity, rgbs]

            # Render
            with torch.no_grad():
                image, _, _, _ = renderer.render(
                    camera.world_to_camera_transform.unsqueeze(0),
                    camera.intrinsics.unsqueeze(0),
                    (camera.image_width, camera.image_height),
                    camera.z_near,
                    camera.z_far,
                    gs_data
                )

            # Convert to PIL Image
            image_np = image[0].cpu().numpy()
            image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)

            images.append(pil_image)

            # CRITICAL: Clean up GPU memory after each view to prevent OOM
            del image, image_np, gs_data, means3D, rotations, scales, opacity, rgbs, features
            del camera  # Free camera tensors too

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.debug(f"  View {i+1}/{num_views} rendered successfully, GPU cache cleared")

        return images

    except Exception as e:
        logger.error(f"Failed to render GaussianModel: {e}")
        return None
