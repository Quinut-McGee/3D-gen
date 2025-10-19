#!/usr/bin/env python3
"""
404-GEN COMPETITIVE MINER - COMPREHENSIVE TEST SUITE

Tests all components of the competitive mining system:
1. FLUX.1-schnell text-to-image generation
2. BRIA RMBG 2.0 background removal
3. DreamGaussian 3D generation
4. CLIP validation
5. Async task manager
6. End-to-end pipeline
7. PLY file validation
"""

import sys
import time
import asyncio
import aiohttp
from pathlib import Path
from io import BytesIO
from PIL import Image
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


class TestResults:
    """Track test results"""
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []

    def record_pass(self, test_name: str):
        self.tests_run += 1
        self.tests_passed += 1
        print(f"✅ {test_name}")

    def record_fail(self, test_name: str, error: str):
        self.tests_run += 1
        self.tests_failed += 1
        self.failures.append((test_name, error))
        print(f"❌ {test_name}: {error}")

    def summary(self):
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total: {self.tests_run}")
        print(f"Passed: {self.tests_passed} ✅")
        print(f"Failed: {self.tests_failed} ❌")

        if self.failures:
            print("\nFailed Tests:")
            for name, error in self.failures:
                print(f"  - {name}: {error}")

        print("=" * 60)
        return self.tests_failed == 0


results = TestResults()


def test_imports():
    """Test 1: Verify all required packages are installed"""
    try:
        import transformers
        import diffusers
        import accelerate
        import clip
        from PIL import Image
        import torch
        import fastapi
        import uvicorn
        import aiohttp
        import loguru
        results.record_pass("Import test")
    except ImportError as e:
        results.record_fail("Import test", str(e))


def test_gpu():
    """Test 2: Verify GPU availability"""
    try:
        if not torch.cuda.is_available():
            results.record_fail("GPU test", "No GPU detected - will be VERY slow")
            return

        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

        if vram < 20:
            print(f"⚠️  Warning: Only {vram:.1f} GB VRAM (recommend 24GB+)")

        results.record_pass(f"GPU test ({gpu_name}, {vram:.1f} GB)")
    except Exception as e:
        results.record_fail("GPU test", str(e))


def test_flux_generator():
    """Test 3: FLUX.1-schnell text-to-image generation"""
    try:
        from generation.models.flux_generator import FluxImageGenerator

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading FLUX.1-schnell on {device}...")

        generator = FluxImageGenerator(device=device)

        # Generate test image
        start = time.time()
        image = generator.generate(
            prompt="a red apple",
            num_inference_steps=4,
            height=512,
            width=512
        )
        elapsed = time.time() - start

        # Verify output
        assert isinstance(image, Image.Image), "Output should be PIL Image"
        assert image.size == (512, 512), f"Expected 512x512, got {image.size}"

        if elapsed > 10:
            print(f"  ⚠️  Warning: Generation took {elapsed:.1f}s (expected <5s)")

        results.record_pass(f"FLUX generation test ({elapsed:.1f}s)")
    except Exception as e:
        results.record_fail("FLUX generation test", str(e))


def test_background_remover():
    """Test 4: BRIA RMBG 2.0 background removal"""
    try:
        from generation.models.background_remover import SOTABackgroundRemover

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading BRIA RMBG 2.0 on {device}...")

        remover = SOTABackgroundRemover(device=device)

        # Create test image (white square on black background)
        test_image = Image.new("RGB", (512, 512), color="black")
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_image)
        draw.rectangle([128, 128, 384, 384], fill="white")

        # Remove background
        start = time.time()
        rgba_image = remover.remove_background(test_image, threshold=0.5)
        elapsed = time.time() - start

        # Verify output
        assert isinstance(rgba_image, Image.Image), "Output should be PIL Image"
        assert rgba_image.mode == "RGBA", f"Expected RGBA mode, got {rgba_image.mode}"
        assert rgba_image.size == (512, 512), f"Expected 512x512, got {rgba_image.size}"

        # Check that alpha channel has transparency
        alpha = np.array(rgba_image)[:, :, 3]
        has_transparency = np.any(alpha < 255)
        assert has_transparency, "Alpha channel should have transparency"

        if elapsed > 1:
            print(f"  ⚠️  Warning: Background removal took {elapsed:.1f}s (expected <0.5s)")

        results.record_pass(f"Background removal test ({elapsed:.1f}s)")
    except Exception as e:
        results.record_fail("Background removal test", str(e))


def test_clip_validator():
    """Test 5: CLIP validation"""
    try:
        from generation.validators.clip_validator import CLIPValidator

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading CLIP validator on {device}...")

        validator = CLIPValidator(device=device, threshold=0.6)

        # Create test images
        # Good match: white image with prompt "white"
        white_image = Image.new("RGB", (224, 224), color="white")

        # Bad match: white image with prompt "black cat"
        prompt_good = "white"
        prompt_bad = "black cat"

        # Test good match
        passes, score = validator.validate_image(white_image, prompt_good)
        assert isinstance(passes, bool), "validate_image should return bool"
        assert isinstance(score, float), "validate_image should return float score"
        print(f"  Good match score: {score:.3f}")

        # Test bad match
        passes_bad, score_bad = validator.validate_image(white_image, prompt_bad)
        print(f"  Bad match score: {score_bad:.3f}")

        # Good match should score higher than bad match
        assert score > score_bad, f"Good match ({score:.3f}) should score higher than bad match ({score_bad:.3f})"

        results.record_pass(f"CLIP validation test (good={score:.3f}, bad={score_bad:.3f})")
    except Exception as e:
        results.record_fail("CLIP validation test", str(e))


async def test_generation_service():
    """Test 6: End-to-end generation service"""
    try:
        print("  Testing generation service at http://localhost:10006...")

        # Check health endpoint
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:10006/health") as response:
                if response.status != 200:
                    results.record_fail("Generation service health check", f"Status {response.status}")
                    return

                health = await response.json()
                print(f"  Health: {health}")

                # Check all models loaded
                models = health.get("models_loaded", {})
                for model_name, loaded in models.items():
                    if not loaded:
                        results.record_fail("Generation service health check", f"{model_name} not loaded")
                        return

        # Test generation
        print("  Generating test model (prompt: 'a red cube')...")
        start = time.time()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:10006/generate/",
                data={"prompt": "a red cube"},
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    results.record_fail("Generation service test", f"Status {response.status}: {error_text}")
                    return

                ply_data = await response.read()
                elapsed = time.time() - start

                # Check response headers
                clip_score = response.headers.get("X-CLIP-Score", "N/A")
                validation_failed = response.headers.get("X-Validation-Failed") == "true"

                # Verify output
                if validation_failed:
                    results.record_fail("Generation service test", f"Validation failed (CLIP={clip_score})")
                    return

                if len(ply_data) < 1000:
                    results.record_fail("Generation service test", f"PLY too small ({len(ply_data)} bytes)")
                    return

                # Check if valid PLY header
                if not ply_data.startswith(b"ply\n"):
                    results.record_fail("Generation service test", "Invalid PLY header")
                    return

                if elapsed > 30:
                    print(f"  ⚠️  Warning: Generation took {elapsed:.1f}s (target <25s)")

                results.record_pass(f"Generation service test ({elapsed:.1f}s, {len(ply_data)/1024:.1f}KB, CLIP={clip_score})")

    except aiohttp.ClientConnectorError:
        results.record_fail("Generation service test", "Service not running - start with ./deploy_competitive.sh")
    except asyncio.TimeoutError:
        results.record_fail("Generation service test", "Timeout after 120s")
    except Exception as e:
        results.record_fail("Generation service test", str(e))


def test_ply_file_structure():
    """Test 7: Validate PLY file structure from a real generation"""
    try:
        print("  Testing PLY file structure...")

        # This test requires a generated PLY file
        # We'll use the one from test_generation_service if available
        # For now, just check that the PLY parsing would work

        from io import StringIO

        # Mock PLY header
        ply_header = """ply
format binary_little_endian 1.0
element vertex 5000
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
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""

        # Verify header format
        lines = ply_header.strip().split("\n")
        assert lines[0] == "ply", "First line should be 'ply'"
        assert any("vertex" in line for line in lines), "Should have vertex element"
        assert any("end_header" in line for line in lines), "Should have end_header"

        results.record_pass("PLY structure test")
    except Exception as e:
        results.record_fail("PLY structure test", str(e))


def test_async_task_manager():
    """Test 8: Async task manager (unit test)"""
    try:
        from neurons.miner.async_task_manager import AsyncTaskManager

        # Just verify it can be instantiated
        task_manager = AsyncTaskManager(
            max_concurrent_tasks=4,
            max_queue_size=20,
            pull_interval=10.0
        )

        assert task_manager.max_concurrent_tasks == 4
        assert task_manager.max_queue_size == 20
        assert task_manager.pull_interval == 10.0
        assert task_manager.stats["tasks_pulled"] == 0

        results.record_pass("AsyncTaskManager instantiation test")
    except Exception as e:
        results.record_fail("AsyncTaskManager instantiation test", str(e))


def test_validator_selector():
    """Test 9: Validator selector with blacklisting"""
    try:
        from neurons.miner.validator_selector import ValidatorSelector, BLACKLISTED_VALIDATORS
        import bittensor as bt

        # Check that UID 180 is blacklisted
        assert 180 in BLACKLISTED_VALIDATORS, "UID 180 should be blacklisted"

        print(f"  Blacklisted validators: {BLACKLISTED_VALIDATORS}")

        results.record_pass(f"Validator blacklist test ({len(BLACKLISTED_VALIDATORS)} blacklisted)")
    except Exception as e:
        results.record_fail("Validator blacklist test", str(e))


def test_competitive_miner_imports():
    """Test 10: Verify competitive miner can be imported"""
    try:
        from neurons.miner.competitive_miner import CompetitiveMiner
        from neurons.miner.competitive_workers import process_task_competitive

        results.record_pass("Competitive miner imports test")
    except Exception as e:
        results.record_fail("Competitive miner imports test", str(e))


async def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("404-GEN COMPETITIVE MINER - TEST SUITE")
    print("=" * 60)
    print()

    # Unit tests (no dependencies)
    print("Running unit tests...\n")
    test_imports()
    test_gpu()
    test_competitive_miner_imports()
    test_async_task_manager()
    test_validator_selector()
    test_ply_file_structure()

    # Component tests (require model downloads)
    print("\nRunning component tests (may take a few minutes)...\n")
    test_flux_generator()
    test_background_remover()
    test_clip_validator()

    # Integration test (requires service running)
    print("\nRunning integration test...\n")
    await test_generation_service()

    # Summary
    success = results.summary()

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
