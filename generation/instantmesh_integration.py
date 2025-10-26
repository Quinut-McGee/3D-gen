# InstantMesh Integration Code
# This file contains the code to replace DreamGaussian with InstantMesh + 2D color sampling

import io
import base64
import httpx
import trimesh
import time
from loguru import logger

async def generate_with_instantmesh(rgba_image, prompt, mesh_to_gaussian_converter, instantmesh_url="http://localhost:10007"):
    """
    Generate 3D Gaussian Splat using InstantMesh + 2D color sampling.

    Args:
        rgba_image: PIL Image (RGBA) from FLUX → background removal
        prompt: Text prompt for logging/debugging
        mesh_to_gaussian_converter: MeshToGaussianConverter instance
        instantmesh_url: InstantMesh microservice URL

    Returns:
        ply_bytes: Binary PLY data
        gs_model: Cached GaussianModel for validation
        timings: Dict of timing info
    """

    # Step 3: 3D mesh generation with InstantMesh microservice (1-2s)
    t3_start = time.time()
    logger.info("  [3/5] Generating 3D mesh with InstantMesh microservice...")

    try:
        # Convert RGBA image to base64 for HTTP transfer
        buffer = io.BytesIO()
        rgba_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Call InstantMesh microservice
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{instantmesh_url}/generate_mesh",
                json={
                    "image_base64": image_base64,
                    "scale": 1.0,
                    "input_size": 384  # Increased from 320 for better mesh quality (512 max, but 384 is good balance)
                }
            )

            if response.status_code != 200:
                raise RuntimeError(f"InstantMesh service returned status {response.status_code}")

            result = response.json()

        # Decode mesh from base64
        mesh_bytes = base64.b64decode(result["mesh_base64"])
        mesh = trimesh.load(io.BytesIO(mesh_bytes), file_type='ply')

        t3_end = time.time()
        logger.info(f"  ✅ Mesh generation done ({t3_end-t3_start:.2f}s, service: {result['generation_time']:.2f}s)")
        logger.info(f"     Mesh quality: {result['num_vertices']} vertices, {result['num_faces']} faces")

    except Exception as e:
        logger.error(f"InstantMesh microservice call failed: {e}", exc_info=True)
        raise

    # Step 4: Convert mesh to Gaussian Splat with 2D color sampling (0.3-0.5s)
    t4_start = time.time()
    logger.info("  [4/5] Converting mesh to Gaussian Splat with 2D color sampling...")

    try:
        # CRITICAL: Pass rgba_image for 2D color sampling!
        ply_buffer = mesh_to_gaussian_converter.convert(
            mesh,
            rgba_image=rgba_image,  # This enables correct colors!
            num_gaussians=12000  # Can adjust for quality/speed trade-off
        )

        # Get PLY bytes
        ply_bytes = ply_buffer.getvalue()

        # Get cached GaussianModel for validation
        gs_model = mesh_to_gaussian_converter.get_last_model()

        t4_end = time.time()
        logger.info(f"  ✅ Mesh-to-Gaussian conversion done ({t4_end-t4_start:.2f}s)")
        logger.info(f"     Generated {len(ply_bytes)/1024:.1f} KB Gaussian Splat PLY")

    except Exception as e:
        logger.error(f"Mesh-to-Gaussian conversion failed: {e}", exc_info=True)
        raise

    timings = {
        "instantmesh": t3_end - t3_start,
        "mesh_to_gaussian": t4_end - t4_start,
        "total_3d": t4_end - t3_start
    }

    return ply_bytes, gs_model, timings
