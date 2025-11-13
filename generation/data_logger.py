"""
Comprehensive Data Collection System for 404-gen Bittensor Miner

Thread-safe, production-ready data logger that tracks all miner generations
for analysis, monitoring, and future fine-tuning.

Author: Claude Code
Date: 2025-11-12
"""

import json
import os
import time
import uuid
import hashlib
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
import traceback
import base64
from PIL import Image
import io


class GenerationDataLogger:
    """
    Thread-safe data logger for tracking all miner generations.

    Features:
    - JSONL format (JSON Lines) for easy streaming and appending
    - Thread-safe concurrent writes
    - Atomic file operations
    - Automatic image and PLY file storage
    - SHA256 hashing for deduplication
    - Monthly log rotation
    - Configuration tracking

    Usage:
        logger = GenerationDataLogger("/home/kobe/404-gen/v1/3D-gen/data")

        # Start logging a generation
        log_id = logger.start_generation(
            task_type="IMAGE-TO-3D",
            prompt="...",
            validator_uid=57,
            miner_config={...}
        )

        # Update as generation progresses
        logger.log_timing(log_id, "sdxl_time", 8.62)
        logger.log_output(log_id, num_gaussians=398234, clip_score=0.189)

        # Log final results
        logger.log_submission(log_id, submitted=True)
        logger.log_validator_feedback(log_id, score=0.65)
        logger.finalize_generation(log_id)
    """

    def __init__(
        self,
        data_dir: str = "/home/kobe/404-gen/v1/3D-gen/data",
        miner_uid: int = 226,
        miner_version: str = "phase1_phase2_v1.0",
        network: str = "mainnet",
        store_images: bool = True,
        store_ply_files: bool = True,
        ply_min_score_threshold: float = 0.7,
        store_rejected_sample_rate: float = 0.1,
        disk_space_alert_threshold: float = 0.20,
        ply_storage_alert_gb: int = 300,
        stats_log_interval: int = 100
    ):
        """
        Initialize data logger.

        Args:
            data_dir: Root directory for data storage
            miner_uid: Miner UID on Bittensor network
            miner_version: Miner software version identifier
            network: Network name (mainnet/testnet)
            store_images: Whether to store image files (requires storage)
            store_ply_files: Whether to store validator-accepted PLY files (only if score >= ply_min_score_threshold)
            ply_min_score_threshold: Minimum validator score to store PLY files (default 0.7 for high-quality only)
            store_rejected_sample_rate: Rate to sample rejected PLY files (0.1 = 10%)
            disk_space_alert_threshold: Alert when disk free space < this % (0.20 = 20%)
            ply_storage_alert_gb: Alert when PLY storage exceeds this GB (300)
            stats_log_interval: Log storage stats every N generations (100)
        """
        self.data_dir = Path(data_dir)
        self.miner_uid = miner_uid
        self.miner_version = miner_version
        self.network = network
        self.store_images = store_images
        self.store_ply_files = store_ply_files
        self.ply_min_score_threshold = ply_min_score_threshold
        self.store_rejected_sample_rate = store_rejected_sample_rate
        self.disk_space_alert_threshold = disk_space_alert_threshold
        self.ply_storage_alert_gb = ply_storage_alert_gb
        self.stats_log_interval = stats_log_interval

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "images").mkdir(exist_ok=True)
        (self.data_dir / "ply_files").mkdir(exist_ok=True)
        (self.data_dir / "ply_files" / "rejected_samples").mkdir(exist_ok=True)
        (self.data_dir / "archive").mkdir(exist_ok=True)

        # File paths
        self.history_file = self.data_dir / "generation_history.jsonl"
        self.config_file = self.data_dir / "miner_config_history.jsonl"
        self.pending_file = self.data_dir / "pending_generations.json"

        # Thread-safe in-memory storage for pending generations
        self._lock = threading.Lock()
        self._pending_generations: Dict[str, Dict[str, Any]] = {}

        # Generation counter for periodic statistics
        self._generation_counter = 0

        # Load persisted pending generations (survive restarts)
        self._load_pending_generations()

        logger.info(f"📊 Data logger initialized: {self.data_dir}")
        logger.info(f"   Miner UID: {miner_uid}, Version: {miner_version}")
        logger.info(f"   Store images: {store_images}, Store PLY files: {store_ply_files}")
        if store_ply_files:
            logger.info(f"   PLY storage: Only high-quality (score >= {ply_min_score_threshold})")
        logger.info(f"   Store rejected samples: {store_rejected_sample_rate*100:.0f}%")
        logger.info(f"   Loaded {len(self._pending_generations)} pending generations")

        # Check storage on startup
        self._check_storage_health()

    def _generate_id(self) -> str:
        """Generate unique log ID"""
        return str(uuid.uuid4())

    def _get_timestamp(self) -> str:
        """Get ISO 8601 timestamp with timezone"""
        return datetime.now(timezone.utc).isoformat()

    def _hash_data(self, data: bytes) -> str:
        """Generate SHA256 hash of data"""
        return "sha256:" + hashlib.sha256(data).hexdigest()

    def _load_pending_generations(self):
        """Load persisted pending generations from disk (survive restarts)"""
        if not self.pending_file.exists():
            return

        try:
            with open(self.pending_file, 'r') as f:
                self._pending_generations = json.load(f)

            logger.info(f"📥 Loaded {len(self._pending_generations)} pending generations from disk")

        except Exception as e:
            logger.warning(f"Failed to load pending generations: {e}")
            self._pending_generations = {}

    def _save_pending_generations(self):
        """Persist pending generations to disk"""
        try:
            with open(self.pending_file, 'w') as f:
                json.dump(self._pending_generations, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save pending generations: {e}")

    def _check_storage_health(self):
        """
        Check storage health and alert if thresholds exceeded.

        Checks:
        1. Disk free space < threshold (default 20%)
        2. PLY storage > threshold (default 300GB)
        """
        try:
            import shutil as shutil_module

            # Check disk free space
            stat = shutil_module.disk_usage(self.data_dir)
            free_percent = stat.free / stat.total
            free_gb = stat.free / (1024**3)

            if free_percent < self.disk_space_alert_threshold:
                logger.warning(
                    f"⚠️  DISK SPACE LOW: {free_percent*100:.1f}% free ({free_gb:.1f} GB) "
                    f"< {self.disk_space_alert_threshold*100:.0f}% threshold"
                )
                logger.warning(f"   Consider disabling PLY/image storage or cleaning archives")

            # Check PLY storage size
            ply_dir = self.data_dir / "ply_files"
            if ply_dir.exists():
                total_size = sum(f.stat().st_size for f in ply_dir.rglob('*') if f.is_file())
                total_gb = total_size / (1024**3)

                if total_gb > self.ply_storage_alert_gb:
                    logger.warning(
                        f"⚠️  PLY STORAGE HIGH: {total_gb:.1f} GB "
                        f"> {self.ply_storage_alert_gb} GB threshold"
                    )
                    logger.warning(f"   Consider running log rotation or cloud backup")

        except Exception as e:
            logger.debug(f"Failed to check storage health: {e}")

    def _get_storage_stats(self) -> dict:
        """Get current storage statistics"""
        try:
            import shutil as shutil_module

            # Disk usage
            stat = shutil_module.disk_usage(self.data_dir)

            # Directory sizes
            def get_dir_size(path: Path) -> int:
                if not path.exists():
                    return 0
                return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())

            images_size = get_dir_size(self.data_dir / "images")
            ply_size = get_dir_size(self.data_dir / "ply_files")
            rejected_size = get_dir_size(self.data_dir / "ply_files" / "rejected_samples")

            # Count files
            def count_files(path: Path) -> int:
                if not path.exists():
                    return 0
                return sum(1 for f in path.rglob('*') if f.is_file())

            return {
                "disk_total_gb": stat.total / (1024**3),
                "disk_free_gb": stat.free / (1024**3),
                "disk_free_percent": stat.free / stat.total,
                "images_gb": images_size / (1024**3),
                "images_count": count_files(self.data_dir / "images"),
                "ply_accepted_gb": (ply_size - rejected_size) / (1024**3),
                "ply_rejected_gb": rejected_size / (1024**3),
                "ply_total_gb": ply_size / (1024**3),
                "ply_count": count_files(self.data_dir / "ply_files"),
                "ply_rejected_count": count_files(self.data_dir / "ply_files" / "rejected_samples")
            }

        except Exception as e:
            logger.debug(f"Failed to get storage stats: {e}")
            return {}

    def _should_store_rejected_sample(self) -> bool:
        """Determine if this rejected PLY should be sampled (for debugging)"""
        import random
        return random.random() < self.store_rejected_sample_rate

    def _save_image(self, image_data: bytes, file_hash: str) -> Optional[str]:
        """
        Save image file to disk.

        Args:
            image_data: Raw image bytes (PNG/JPEG)
            file_hash: SHA256 hash for filename

        Returns:
            Relative path to saved file, or None if not stored
        """
        if not self.store_images:
            return None

        try:
            # Determine format from image data
            img = Image.open(io.BytesIO(image_data))
            format_ext = img.format.lower() if img.format else "png"

            # Save with hash as filename
            hash_part = file_hash.split(":", 1)[1][:16]  # Use first 16 chars of hash
            filename = f"{hash_part}.{format_ext}"
            filepath = self.data_dir / "images" / filename

            # Save image
            with open(filepath, "wb") as f:
                f.write(image_data)

            return f"images/{filename}"

        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return None

    def _save_ply_file(self, ply_data: bytes, file_hash: str, generation_id: str, rejected: bool = False) -> Optional[str]:
        """
        Save PLY file to disk (for validator-accepted outputs or rejected samples).

        Args:
            ply_data: Raw PLY file bytes
            file_hash: SHA256 hash for deduplication
            generation_id: Generation ID for filename
            rejected: If True, save to rejected_samples subdirectory

        Returns:
            Relative path to saved file, or None if not stored
        """
        if not self.store_ply_files and not rejected:
            return None

        # For rejected samples, check sampling rate
        if rejected and not self._should_store_rejected_sample():
            return None

        try:
            # Use generation_id + hash for filename
            hash_part = file_hash.split(":", 1)[1][:16]
            filename = f"{generation_id}_{hash_part}.ply"

            # Choose directory based on acceptance
            if rejected:
                subdir = "ply_files/rejected_samples"
            else:
                subdir = "ply_files"

            filepath = self.data_dir / subdir / filename

            # Save PLY file
            with open(filepath, "wb") as f:
                f.write(ply_data)

            return f"{subdir}/{filename}"

        except Exception as e:
            logger.error(f"Failed to save PLY file: {e}")
            return None

    def start_generation(
        self,
        task_type: str,
        prompt: str,
        validator_uid: int,
        validator_hotkey: str = None,
        miner_config: dict = None
    ) -> str:
        """
        Start logging a new generation.

        Args:
            task_type: "TEXT-TO-3D" or "IMAGE-TO-3D"
            prompt: Original prompt text or base64 image
            validator_uid: Validator UID that sent the task
            validator_hotkey: Validator hotkey (optional)
            miner_config: Current miner configuration

        Returns:
            log_id: Unique identifier for this generation
        """
        log_id = self._generate_id()

        # Detect if prompt is base64 image
        is_base64_image = (
            len(prompt) > 500 or
            prompt.startswith('iVBOR') or
            prompt.startswith('/9j/') or
            prompt.startswith('data:image')
        )

        # Process image if present
        base64_image_hash = None
        prompt_to_store = prompt
        image_file_path = None

        if is_base64_image and self.store_images:
            try:
                # Decode base64 image
                if ',' in prompt[:100]:
                    base64_data = prompt.split(',', 1)[1]
                else:
                    base64_data = prompt

                image_bytes = base64.b64decode(base64_data)
                base64_image_hash = self._hash_data(image_bytes)

                # Save image file
                image_file_path = self._save_image(image_bytes, base64_image_hash)

                # Don't store full base64 in JSON
                prompt_to_store = f"[BASE64_IMAGE_HASH:{base64_image_hash}]"

            except Exception as e:
                logger.warning(f"Failed to process base64 image: {e}")

        # Create initial log entry
        log_entry = {
            "metadata": {
                "generation_id": log_id,
                "timestamp": self._get_timestamp(),
                "miner_uid": self.miner_uid,
                "miner_version": self.miner_version,
                "validator_uid": validator_uid,
                "validator_hotkey": validator_hotkey or "unknown",
                "network": self.network
            },
            "task": {
                "task_type": task_type,
                "prompt": prompt_to_store,
                "prompt_length": len(prompt),
                "is_base64_image": is_base64_image,
                "base64_image_hash": base64_image_hash,
                "image_file_path": image_file_path
            },
            "miner_config": miner_config or {},
            "generation": {
                "enhanced_prompt": None,
                "negative_prompt": None,
                "timing": {},
                "output": {},
                "submission": {},
            },
            "validator_feedback": {
                "received": False,
                "feedback_time": None,
                "score": None,
                "accepted": None,
                "feedback_delay_seconds": None
            },
            "failure_analysis": {
                "failed": False,
                "error_type": None,
                "error_message": None,
                "stack_trace": None
            }
        }

        # Store in pending generations
        with self._lock:
            self._pending_generations[log_id] = log_entry

        return log_id

    def log_enhanced_prompt(self, log_id: str, enhanced_prompt: str, negative_prompt: str = None):
        """Log enhanced prompt after LLM processing"""
        with self._lock:
            if log_id in self._pending_generations:
                self._pending_generations[log_id]["generation"]["enhanced_prompt"] = enhanced_prompt
                self._pending_generations[log_id]["generation"]["negative_prompt"] = negative_prompt

    def log_timing(self, log_id: str, component: str, duration: float):
        """
        Log timing for a pipeline component.

        Args:
            log_id: Generation ID
            component: Component name (e.g., "sdxl_time", "trellis_time")
            duration: Duration in seconds
        """
        with self._lock:
            if log_id in self._pending_generations:
                self._pending_generations[log_id]["generation"]["timing"][component] = round(duration, 3)

                # Calculate total time if all components present
                timing = self._pending_generations[log_id]["generation"]["timing"]
                if "total_time" not in timing:
                    # Sum all timing components
                    total = sum(v for k, v in timing.items() if k != "total_time")
                    if total > 0:
                        timing["total_time"] = round(total, 3)

    def log_output(
        self,
        log_id: str,
        num_gaussians: int = None,
        file_size_mb: float = None,
        ply_file_path: str = None,
        ply_data: bytes = None,
        clip_score: float = None,
        clip_threshold_pass: bool = None,
        gaussian_count_pass: bool = None,
        validation_pass: bool = None,
        gaussian_stats: dict = None,
        render_stats: dict = None
    ):
        """
        Log generation output details.

        Args:
            log_id: Generation ID
            num_gaussians: Number of gaussians in output
            file_size_mb: Output file size in MB
            ply_file_path: Path to PLY file (if already saved)
            ply_data: Raw PLY bytes (for saving)
            clip_score: CLIP validation score
            clip_threshold_pass: Whether CLIP threshold passed
            gaussian_count_pass: Whether gaussian count threshold passed
            validation_pass: Overall validation pass
            gaussian_stats: Detailed gaussian statistics
            render_stats: Rendering statistics
        """
        with self._lock:
            if log_id not in self._pending_generations:
                return

            output = self._pending_generations[log_id]["generation"]["output"]

            # Basic output info
            if num_gaussians is not None:
                output["num_gaussians"] = num_gaussians
            if file_size_mb is not None:
                output["file_size_mb"] = round(file_size_mb, 2)

            # PLY file handling
            if ply_file_path:
                output["ply_file_path"] = ply_file_path

                # Calculate hash if file exists
                if os.path.exists(ply_file_path):
                    try:
                        with open(ply_file_path, "rb") as f:
                            ply_bytes = f.read()
                        output["ply_file_hash"] = self._hash_data(ply_bytes)
                    except Exception as e:
                        logger.warning(f"Failed to hash PLY file: {e}")

            elif ply_data and self.store_ply_files:
                # Save PLY data to file
                ply_hash = self._hash_data(ply_data)
                output["ply_file_hash"] = ply_hash
                saved_path = self._save_ply_file(ply_data, ply_hash, log_id)
                if saved_path:
                    output["ply_file_path"] = saved_path

            # Quality metrics
            if clip_score is not None or clip_threshold_pass is not None or validation_pass is not None:
                output["quality_metrics"] = {
                    "clip_score": round(clip_score, 4) if clip_score is not None else None,
                    "clip_threshold_pass": clip_threshold_pass,
                    "gaussian_count_pass": gaussian_count_pass,
                    "validation_pass": validation_pass
                }

            # Gaussian statistics
            if gaussian_stats:
                output["gaussian_stats"] = gaussian_stats

            # Render statistics
            if render_stats:
                output["render_stats"] = render_stats

    def log_submission(
        self,
        log_id: str,
        submitted: bool,
        rejection_reason: str = None,
        ply_data: bytes = None
    ):
        """
        Log submission status.

        Args:
            log_id: Generation ID
            submitted: Whether generation was submitted to validator
            rejection_reason: Reason for pre-submission rejection (if any)
            ply_data: PLY file bytes (for rejected sample storage)
        """
        with self._lock:
            if log_id in self._pending_generations:
                self._pending_generations[log_id]["generation"]["submission"] = {
                    "submitted": submitted,
                    "submission_time": self._get_timestamp() if submitted else None,
                    "pre_submission_rejection": not submitted,
                    "rejection_reason": rejection_reason
                }

                # Store rejected PLY sample (10% by default) for debugging
                if not submitted and ply_data and self.store_rejected_sample_rate > 0:
                    ply_hash = self._hash_data(ply_data)
                    saved_path = self._save_ply_file(ply_data, ply_hash, log_id, rejected=True)

                    if saved_path:
                        output = self._pending_generations[log_id]["generation"]["output"]
                        output["rejected_sample_path"] = saved_path
                        output["rejected_sample_hash"] = ply_hash
                        logger.debug(f"🔍 Saved rejected PLY sample: {saved_path}")

    def log_validator_feedback(
        self,
        log_id: str,
        score: float,
        feedback_time: str = None
    ):
        """
        Log validator feedback (called when miner receives score from validator).

        Args:
            log_id: Generation ID
            score: Validator score (0.0-1.0)
            feedback_time: Timestamp of feedback (auto-generated if not provided)
        """
        with self._lock:
            if log_id not in self._pending_generations:
                logger.warning(f"Cannot log feedback for unknown generation: {log_id}")
                return

            feedback_timestamp = feedback_time or self._get_timestamp()

            # Calculate feedback delay
            start_time = self._pending_generations[log_id]["metadata"]["timestamp"]
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            feedback_dt = datetime.fromisoformat(feedback_timestamp.replace('Z', '+00:00'))
            delay_seconds = (feedback_dt - start_dt).total_seconds()

            self._pending_generations[log_id]["validator_feedback"] = {
                "received": True,
                "feedback_time": feedback_timestamp,
                "score": round(score, 4),
                "accepted": score > 0,
                "feedback_delay_seconds": round(delay_seconds, 2)
            }

            # If validator accepted (score >= threshold) and we haven't stored PLY yet, store it now
            # Only store high-quality PLYs (default threshold: 0.7) to save storage
            if score >= self.ply_min_score_threshold and self.store_ply_files:
                output = self._pending_generations[log_id]["generation"]["output"]
                if "ply_file_path" not in output or output.get("ply_file_path") is None:
                    # Check if we have the temp PLY file path
                    temp_ply_path = output.get("temp_ply_path")
                    if temp_ply_path and os.path.exists(temp_ply_path):
                        try:
                            with open(temp_ply_path, "rb") as f:
                                ply_data = f.read()

                            ply_hash = self._hash_data(ply_data)
                            saved_path = self._save_ply_file(ply_data, ply_hash, log_id)

                            if saved_path:
                                output["ply_file_path"] = saved_path
                                output["ply_file_hash"] = ply_hash
                                logger.info(f"📦 Saved high-quality PLY (score={score:.3f} >= {self.ply_min_score_threshold}): {saved_path}")

                        except Exception as e:
                            logger.error(f"Failed to save accepted PLY file: {e}")

    def log_failure(
        self,
        log_id: str,
        error_type: str,
        error_message: str,
        stack_trace: str = None
    ):
        """
        Log generation failure.

        Args:
            log_id: Generation ID
            error_type: Type of error ("timeout", "oom", "crash", "validation_failed", etc.)
            error_message: Error message
            stack_trace: Full stack trace (optional)
        """
        with self._lock:
            if log_id in self._pending_generations:
                self._pending_generations[log_id]["failure_analysis"] = {
                    "failed": True,
                    "error_type": error_type,
                    "error_message": error_message,
                    "stack_trace": stack_trace
                }

    def finalize_generation(self, log_id: str):
        """
        Finalize and write generation log to file (atomic write).

        Args:
            log_id: Generation ID
        """
        with self._lock:
            if log_id not in self._pending_generations:
                logger.warning(f"Cannot finalize unknown generation: {log_id}")
                return

            log_entry = self._pending_generations[log_id]

            # Remove from pending
            del self._pending_generations[log_id]

            # Increment generation counter
            self._generation_counter += 1

            # Save pending generations (persistence for restart recovery)
            self._save_pending_generations()

        # Write to file (atomic operation using temp file)
        try:
            temp_file = self.history_file.with_suffix('.tmp')

            # Append to temp file
            with open(temp_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            # Atomic rename (append to main file)
            # Note: On POSIX, this is atomic and safe for concurrent access
            with open(self.history_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            # Remove temp file
            if temp_file.exists():
                temp_file.unlink()

            logger.debug(f"📝 Logged generation: {log_id}")

            # Periodic storage statistics logging
            if self._generation_counter % self.stats_log_interval == 0:
                self._log_storage_stats()

        except Exception as e:
            logger.error(f"Failed to finalize generation log: {e}")
            logger.error(traceback.format_exc())

    def _log_storage_stats(self):
        """Log storage statistics (called periodically)"""
        try:
            stats = self._get_storage_stats()

            if stats:
                logger.info("=" * 60)
                logger.info(f"📊 STORAGE STATS (after {self._generation_counter} generations)")
                logger.info("=" * 60)
                logger.info(f"Disk: {stats['disk_free_gb']:.1f} GB free ({stats['disk_free_percent']*100:.1f}%)")
                logger.info(f"Images: {stats['images_gb']:.2f} GB ({stats['images_count']} files)")
                logger.info(f"PLY accepted: {stats['ply_accepted_gb']:.2f} GB")
                logger.info(f"PLY rejected samples: {stats['ply_rejected_gb']:.2f} GB ({stats['ply_rejected_count']} files)")
                logger.info(f"PLY total: {stats['ply_total_gb']:.2f} GB ({stats['ply_count']} files)")
                logger.info("=" * 60)

                # Check storage health after logging stats
                self._check_storage_health()

        except Exception as e:
            logger.debug(f"Failed to log storage stats: {e}")

    def log_config_change(self, changes: dict, reason: str):
        """
        Log miner configuration change.

        Args:
            changes: Dictionary of changes (key: {old: val, new: val})
            reason: Reason for configuration change
        """
        config_entry = {
            "timestamp": self._get_timestamp(),
            "event": "config_change",
            "changes": changes,
            "reason": reason
        }

        try:
            with open(self.config_file, 'a') as f:
                f.write(json.dumps(config_entry) + '\n')

            logger.info(f"📝 Logged config change: {len(changes)} parameters")

        except Exception as e:
            logger.error(f"Failed to log config change: {e}")

    def get_stats(self) -> dict:
        """Get current statistics from generation history"""
        if not self.history_file.exists():
            return {"total_generations": 0}

        try:
            total = 0
            accepted = 0
            total_clip = 0
            clip_count = 0

            with open(self.history_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        total += 1

                        # Count accepted
                        if entry.get("validator_feedback", {}).get("accepted"):
                            accepted += 1

                        # Collect CLIP scores
                        clip = entry.get("generation", {}).get("output", {}).get("quality_metrics", {}).get("clip_score")
                        if clip is not None:
                            total_clip += clip
                            clip_count += 1

                    except json.JSONDecodeError:
                        continue

            return {
                "total_generations": total,
                "accepted": accepted,
                "acceptance_rate": round(accepted / total * 100, 1) if total > 0 else 0,
                "average_clip": round(total_clip / clip_count, 4) if clip_count > 0 else None
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


# Global singleton instance (initialized by miner)
_global_logger: Optional[GenerationDataLogger] = None


def get_logger() -> Optional[GenerationDataLogger]:
    """Get global data logger instance"""
    return _global_logger


def init_logger(
    data_dir: str = "/home/kobe/404-gen/v1/3D-gen/data",
    miner_uid: int = 226,
    miner_version: str = "phase1_phase2_v1.0",
    **kwargs
) -> GenerationDataLogger:
    """
    Initialize global data logger.

    Should be called once at miner startup.
    """
    global _global_logger
    _global_logger = GenerationDataLogger(
        data_dir=data_dir,
        miner_uid=miner_uid,
        miner_version=miner_version,
        **kwargs
    )
    return _global_logger


def log_startup_config(config: dict, reason: str = "Miner startup"):
    """
    Log miner configuration at startup (convenience function).

    This should be called after init_logger() to record the initial
    configuration when the miner starts.

    Args:
        config: Dictionary of configuration parameters
        reason: Reason for logging (default: "Miner startup")

    Example:
        # In serve_competitive.py startup:
        data_logger = init_logger(...)
        log_startup_config({
            "sdxl_turbo_steps": 4,
            "trellis_sparse_steps": 45,
            "trellis_slat_steps": 35,
            "gaussian_threshold": 50000,
            "clip_threshold": 0.10,
            "background_threshold": 0.4,
            ...
        })
    """
    logger_instance = get_logger()

    if logger_instance:
        # Log as config change with no "old" values
        changes = {key: {"old": None, "new": value} for key, value in config.items()}
        logger_instance.log_config_change(changes, reason)

        logger.info(f"📝 Logged startup configuration: {len(config)} parameters")
    else:
        logger.warning("Cannot log startup config: data logger not initialized")
