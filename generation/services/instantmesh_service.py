"""
InstantMesh Microservice with LAZY LOADING
Runs in isolated 'instantmesh' conda environment to avoid dependency conflicts.

VRAM OPTIMIZATION STRATEGY:
- Idle state: ~500MB (just Python/FastAPI running)
- During generation: Load model to GPU → generate mesh (1.7s) → offload to CPU → clear cache
- Post-generation: Back to ~500MB

This allows FLUX and InstantMesh to share the RTX 4090 sequentially instead of in parallel,
staying within the 24GB VRAM limit.
"""

import sys
import os
import io
import base64
import logging
import gc
from typing import Optional

# Ensure ninja is in PATH before importing anything that needs CUDA extensions
conda_bin = '/home/kobe/miniconda3/envs/instantmesh/bin'
if conda_bin not in os.environ.get('PATH', ''):
    os.environ['PATH'] = f"{conda_bin}:{os.environ.get('PATH', '')}"

# Add InstantMesh to path
sys.path.insert(0, '/tmp/InstantMesh')

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import trimesh

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import get_zero123plus_input_cameras
from omegaconf import OmegaConf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="InstantMesh Service (Lazy Loading)", version="2.0")

# Global model storage - LAZY LOADED
class ModelState:
    model = None  # Will be loaded on-demand
    config = None
    model_path = "/tmp/InstantMesh/ckpts/instant_mesh_large.ckpt"
    config_path = "/tmp/InstantMesh/configs/instant-mesh-large.yaml"
    device = None
    is_model_on_gpu = False  # Track if model is currently on GPU

state = ModelState()


# Request/Response models
class GenerateMeshRequest(BaseModel):
    image_base64: str  # Base64 encoded RGBA PNG
    scale: float = 1.0
    input_size: int = 320


class GenerateMeshResponse(BaseModel):
    mesh_base64: str  # Base64 encoded PLY file
    num_vertices: int
    num_faces: int
    generation_time: float


@app.on_event("startup")
async def startup():
    """Minimal startup - just initialize config, don't load model"""
    try:
        logger.info("=" * 80)
        logger.info("INSTANTMESH SERVICE STARTING (LAZY LOADING MODE)")
        logger.info("=" * 80)

        state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {state.device}")

        # Load config only (lightweight)
        state.config = OmegaConf.load(state.config_path)
        logger.info(f"✅ Config loaded from {state.config_path}")

        logger.info("=" * 80)
        logger.info("✅ SERVICE READY (Model will load on first request)")
        logger.info("   Idle VRAM usage: ~500MB")
        logger.info("   During generation: ~23GB (1.7s), then offloads")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Failed to initialize service: {e}", exc_info=True)
        raise


def load_model_to_gpu():
    """Load model to GPU on-demand"""
    if state.model is None:
        logger.info("🔄 Loading InstantMesh model for first time...")

        model_config = state.config.model_config
        state.model = instantiate_from_config(model_config)

        # Load weights to CPU first
        checkpoint = torch.load(state.model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        # Remove 'lrm_generator.' prefix from keys
        state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
        state.model.load_state_dict(state_dict, strict=True)
        logger.info(f"✅ Loaded weights from {state.model_path}")

    # Move to GPU if not already there
    if not state.is_model_on_gpu:
        logger.info("🔄 Moving model to GPU...")
        state.model = state.model.to(state.device)
        state.model.init_flexicubes_geometry(state.device, fovy=30.0)
        state.model = state.model.eval()
        state.is_model_on_gpu = True
        logger.info("✅ Model on GPU, ready for generation")


def offload_model_from_gpu():
    """Offload model to CPU and clear CUDA cache to free VRAM"""
    if state.model is not None and state.is_model_on_gpu:
        logger.info("🔄 Offloading model from GPU to CPU...")
        state.model = state.model.cpu()
        state.is_model_on_gpu = False

        # Aggressive VRAM cleanup
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        logger.info("✅ Model offloaded, VRAM freed")


def preprocess_image(image: Image.Image, target_size: int = 320) -> torch.Tensor:
    """Preprocess input image for InstantMesh"""
    # Convert RGBA to RGB with white background if needed
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to target size
    image = image.resize((target_size, target_size), Image.LANCZOS)

    # Convert to tensor and normalize to [0, 1]
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    return image_tensor


def extract_mesh_from_output(mesh_output: tuple, scale: float = 1.0) -> trimesh.Trimesh:
    """Extract trimesh from InstantMesh model output

    Args:
        mesh_output: Tuple of (vertices, faces, vertex_colors) from extract_mesh
        scale: Scale factor for vertices

    Returns:
        trimesh.Trimesh object
    """
    # Unpack the tuple returned by extract_mesh
    vertices, faces, vertex_colors = mesh_output

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


@app.post("/generate_mesh", response_model=GenerateMeshResponse)
async def generate_mesh(request: GenerateMeshRequest):
    """
    Generate 3D mesh from RGBA image with LAZY LOADING.

    Model is loaded to GPU only during generation, then immediately offloaded.
    Target performance: 1.7s generation + ~2s load/offload overhead on first call
    """
    try:
        import time
        start_time = time.time()

        # STEP 1: Load model to GPU (only happens when needed)
        load_start = time.time()
        load_model_to_gpu()
        load_time = time.time() - load_start
        if load_time > 0.1:
            logger.info(f"   Model load time: {load_time:.2f}s")

        # Decode base64 image
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes))

        logger.info(f"Generating mesh from {image.size} {image.mode} image...")

        # Preprocess image
        image_tensor = preprocess_image(image, target_size=request.input_size)
        image_tensor = image_tensor.to(state.device)

        # InstantMesh expects multi-view input [B, V, C, H, W]
        # Since we have single image, replicate it 6 times for 6 views
        image_tensor = image_tensor.unsqueeze(1).repeat(1, 6, 1, 1, 1)  # [1, 6, 3, H, W]

        # Get camera parameters for 6 views
        cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0, fov=30.0)
        cameras = cameras.to(state.device)

        # STEP 2: Generate mesh
        gen_start = time.time()
        with torch.no_grad():
            # Forward pass to get triplane features
            planes = state.model.forward_planes(image_tensor, cameras)

            # Extract mesh using FlexiCubes
            mesh_output = state.model.extract_mesh(
                planes,
                use_texture_map=False,  # Faster without texture map
            )
        gen_time = time.time() - gen_start
        logger.info(f"   Mesh generation: {gen_time:.2f}s")

        # Convert to trimesh
        mesh = extract_mesh_from_output(mesh_output, scale=request.scale)

        # Export mesh to PLY bytes
        ply_buffer = io.BytesIO()
        mesh.export(ply_buffer, file_type='ply', encoding='binary')
        ply_bytes = ply_buffer.getvalue()

        # Encode to base64
        mesh_base64 = base64.b64encode(ply_bytes).decode('utf-8')

        # STEP 3: Offload model from GPU to free VRAM
        offload_start = time.time()
        offload_model_from_gpu()
        offload_time = time.time() - offload_start
        logger.info(f"   Model offload time: {offload_time:.2f}s")

        generation_time = time.time() - start_time

        logger.info(f"✅ Total time: {generation_time:.2f}s ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)")

        return GenerateMeshResponse(
            mesh_base64=mesh_base64,
            num_vertices=len(mesh.vertices),
            num_faces=len(mesh.faces),
            generation_time=generation_time
        )

    except Exception as e:
        logger.error(f"❌ Mesh generation failed: {e}", exc_info=True)
        # Try to offload even on error
        try:
            offload_model_from_gpu()
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "lazy_loading": True,
        "model_on_gpu": state.is_model_on_gpu,
        "device": str(state.device) if state.device else None
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "InstantMesh Microservice (Lazy Loading)",
        "version": "2.0",
        "status": "ready",
        "lazy_loading": True,
        "model_on_gpu": state.is_model_on_gpu,
        "endpoints": {
            "generate_mesh": "/generate_mesh",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=10007,
        log_level="info",
        access_log=True
    )
