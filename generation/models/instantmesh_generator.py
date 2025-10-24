"""
InstantMesh Generator Module
Wraps the InstantMesh model for fast single-image to 3D mesh generation.
Target: 6-8 seconds for mesh generation (compared to DreamGaussian's 28s)
"""

import sys
import os
import torch
import numpy as np
import trimesh
from PIL import Image
from typing import Optional
from omegaconf import OmegaConf
import logging

# Add InstantMesh to path
sys.path.insert(0, '/tmp/InstantMesh')

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import FOV_to_intrinsics, get_circular_camera_poses
from src.utils.mesh_util import save_obj

logger = logging.getLogger(__name__)


class InstantMeshGenerator:
    """
    Wrapper for InstantMesh model with lazy loading and GPU memory management.

    Performance target: 6-8 seconds per generation
    Memory management: Can move to CPU when not in use
    """

    def __init__(
        self,
        device: str = "cuda",
        model_path: str = "/tmp/InstantMesh/ckpts/instant_mesh_large.ckpt",
        config_path: str = "/tmp/InstantMesh/configs/instant-mesh-large.yaml",
    ):
        """
        Initialize InstantMesh generator with lazy loading.

        Args:
            device: Device to run model on ('cuda' or 'cpu')
            model_path: Path to InstantMesh checkpoint
            config_path: Path to InstantMesh config file
        """
        self.device = torch.device(device)
        self.model_path = model_path
        self.config_path = config_path

        self.model = None
        self.config = None
        self._is_initialized = False

        logger.info(f"InstantMeshGenerator initialized (lazy loading enabled)")

    def _load_model(self):
        """
        Lazy load the model when first needed.
        This avoids startup time if model isn't used.
        """
        if self._is_initialized:
            return

        logger.info("Loading InstantMesh model...")
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

        # Load config
        self.config = OmegaConf.load(self.config_path)
        model_config = self.config.model_config

        # Instantiate model
        self.model = instantiate_from_config(model_config)

        # Load weights
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location='cpu')['state_dict']
            # Remove 'lrm_generator.' prefix from keys
            state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
            self.model.load_state_dict(state_dict, strict=True)
            logger.info(f"Loaded weights from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        # Move to device and initialize geometry
        self.model = self.model.to(self.device)
        self.model.init_flexicubes_geometry(self.device, fovy=30.0)
        self.model = self.model.eval()

        self._is_initialized = True

        end_time.record()
        torch.cuda.synchronize()
        load_time = start_time.elapsed_time(end_time) / 1000.0
        logger.info(f"InstantMesh model loaded in {load_time:.2f}s")

    def _preprocess_image(self, image: Image.Image, target_size: int = 320) -> torch.Tensor:
        """
        Preprocess input image for InstantMesh.

        Args:
            image: PIL Image (RGBA or RGB)
            target_size: Target size for model input

        Returns:
            Preprocessed image tensor [1, 3, H, W]
        """
        # Convert RGBA to RGB with white background if needed
        if image.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to target size
        image = image.resize((target_size, target_size), Image.LANCZOS)

        # Convert to tensor and normalize to [0, 1]
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        return image_tensor

    def _extract_mesh_from_output(self, model_output: dict, scale: float = 1.0) -> trimesh.Trimesh:
        """
        Extract trimesh from InstantMesh model output.

        Args:
            model_output: Output from model.forward_geometry
            scale: Scale factor for mesh

        Returns:
            trimesh.Trimesh object
        """
        # Extract vertices and faces
        vertices = model_output['v_pos'].cpu().numpy()[0]  # [N, 3]
        faces = model_output['t_pos_idx'].cpu().numpy()[0]  # [M, 3]

        # Extract vertex colors if available
        vertex_colors = None
        if 'v_rgb' in model_output:
            vertex_colors = model_output['v_rgb'].cpu().numpy()[0]  # [N, 3]
            # Convert from [0, 1] to [0, 255]
            vertex_colors = (vertex_colors * 255).astype(np.uint8)

        # Apply scale
        vertices = vertices * scale

        # Create trimesh
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            process=False  # Don't process mesh (faster)
        )

        return mesh

    @torch.no_grad()
    def generate_mesh(
        self,
        image: Image.Image,
        scale: float = 1.0,
        input_size: int = 320,
    ) -> trimesh.Trimesh:
        """
        Generate 3D mesh from single input image.

        TARGET PERFORMANCE: 6-8 seconds

        Args:
            image: Input PIL Image (RGB or RGBA)
            scale: Scale factor for output mesh
            input_size: Input image size for model

        Returns:
            trimesh.Trimesh object

        Raises:
            RuntimeError: If generation fails
        """
        # Ensure model is loaded
        self._load_model()

        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

        try:
            # Preprocess image
            image_tensor = self._preprocess_image(image, target_size=input_size)
            image_tensor = image_tensor.to(self.device)

            # Forward pass to get triplane features
            # InstantMesh expects images in shape [B, 3, H, W]
            planes = self.model.forward_planes(image_tensor)

            # Generate mesh using FlexiCubes
            # We need a single camera for mesh extraction (identity camera)
            batch_size = 1
            render_cameras = torch.eye(4).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, 4, 4]

            # Extract mesh
            mesh_output = self.model.extract_mesh(
                planes,
                use_texture_map=False,  # Faster without texture map
            )

            # Convert to trimesh
            mesh = self._extract_mesh_from_output(mesh_output, scale=scale)

            end_time.record()
            torch.cuda.synchronize()
            generation_time = start_time.elapsed_time(end_time) / 1000.0

            logger.info(f"InstantMesh generation completed in {generation_time:.2f}s")
            logger.info(f"Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

            return mesh

        except Exception as e:
            logger.error(f"InstantMesh generation failed: {e}")
            raise RuntimeError(f"Failed to generate mesh: {e}")

    def to_cpu(self):
        """Move model to CPU to free GPU memory."""
        if self.model is not None and self._is_initialized:
            self.model = self.model.cpu()
            torch.cuda.empty_cache()
            logger.info("InstantMesh model moved to CPU")

    def to_gpu(self, device: Optional[str] = None):
        """Move model back to GPU."""
        if self.model is not None and self._is_initialized:
            if device is not None:
                self.device = torch.device(device)
            self.model = self.model.to(self.device)
            logger.info(f"InstantMesh model moved to {self.device}")

    def clear_memory(self):
        """Clear model from memory completely."""
        if self.model is not None:
            del self.model
            self.model = None
            self._is_initialized = False
            torch.cuda.empty_cache()
            logger.info("InstantMesh model cleared from memory")


# Quick test function
if __name__ == "__main__":
    # Test with a simple image
    print("Testing InstantMeshGenerator...")

    # Create a simple test image
    test_image = Image.new('RGB', (512, 512), color=(255, 255, 255))

    # Initialize generator
    generator = InstantMeshGenerator(device="cuda")

    # Generate mesh
    mesh = generator.generate_mesh(test_image)

    print(f"✅ Test successful!")
    print(f"Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
