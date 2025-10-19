"""
CLIP-based validation system for pre-submission quality checking.
Prevents cooldown penalties from low-quality submissions.
"""

import torch
import clip
from PIL import Image
import io
from typing import Tuple
from loguru import logger
import numpy as np


class CLIPValidator:
    """
    Validates 3D model renders against text prompts using CLIP.

    This is CRITICAL for competitive mining:
    - Low quality results (CLIP < 0.6) cause cooldown penalties
    - Empty results are ignored (better than penalties)
    - Pre-validation ensures only good results are submitted
    """

    def __init__(
        self,
        device: str = "cuda",
        threshold: float = 0.6,
        model_name: str = "ViT-B/32"
    ):
        """
        Args:
            device: CUDA device for inference
            threshold: Minimum CLIP score to pass (network uses 0.6)
            model_name: CLIP model variant (ViT-B/32 is fast and accurate)
        """
        self.device = device
        self.threshold = threshold

        logger.info(f"Loading CLIP model {model_name} for validation...")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()  # Inference mode

        logger.info(f"CLIP validator ready. Threshold: {threshold}")

    @torch.no_grad()
    def validate_image(
        self,
        image: Image.Image,
        prompt: str
    ) -> Tuple[bool, float]:
        """
        Validate a rendered image against text prompt.

        Args:
            image: PIL Image of the 3D model render
            prompt: Original text prompt

        Returns:
            (passes_threshold, clip_score)

        Example:
            >>> validator = CLIPValidator()
            >>> passes, score = validator.validate_image(image, "a red car")
            >>> if not passes:
            >>>     logger.warning(f"Failed validation: {score:.3f} < 0.6")
        """
        try:
            # Preprocess image
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            # Tokenize text
            text_input = clip.tokenize([prompt]).to(self.device)

            # Get embeddings
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Cosine similarity (CLIP score)
            similarity = (image_features @ text_features.T).item()

            # Check threshold
            passes = similarity >= self.threshold

            return passes, similarity

        except Exception as e:
            logger.error(f"CLIP validation error: {e}")
            # On error, be conservative and reject
            return False, 0.0

    @torch.no_grad()
    def validate_ply_renders(
        self,
        ply_bytes: bytes,
        prompt: str,
        num_views: int = 4
    ) -> Tuple[bool, float]:
        """
        Validate a PLY file by rendering it and checking CLIP scores.

        This is the main validation method for production use.
        Renders the PLY from multiple angles and uses the average CLIP score.

        Args:
            ply_bytes: Raw PLY file bytes
            prompt: Original text prompt
            num_views: Number of views to render for validation

        Returns:
            (passes_threshold, average_clip_score)
        """
        try:
            # Import here to avoid circular dependency
            from ..rendering.quick_render import render_ply_to_images

            # Render PLY to images
            images = render_ply_to_images(ply_bytes, num_views=num_views)

            if not images:
                logger.warning("Failed to render PLY for validation")
                return False, 0.0

            # Validate each view
            scores = []
            for i, image in enumerate(images):
                _, score = self.validate_image(image, prompt)
                scores.append(score)
                logger.debug(f"View {i+1}/{num_views}: CLIP={score:.3f}")

            # Use average score
            avg_score = np.mean(scores)
            passes = avg_score >= self.threshold

            logger.info(
                f"Validation result: {avg_score:.3f} "
                f"({'PASS' if passes else 'FAIL'}, threshold={self.threshold})"
            )

            return passes, avg_score

        except Exception as e:
            logger.error(f"PLY validation error: {e}")
            return False, 0.0

    def set_threshold(self, new_threshold: float):
        """
        Update validation threshold.

        The network's threshold increases over time as models improve.
        This allows dynamic adjustment.
        """
        logger.info(f"Updating CLIP threshold: {self.threshold} → {new_threshold}")
        self.threshold = new_threshold
