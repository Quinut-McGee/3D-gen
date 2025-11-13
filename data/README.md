# 404-GEN Miner Data Collection System

Comprehensive data collection system for tracking all miner generations on Bittensor Subnet 17.

## Directory Structure

```
data/
├── generation_history.jsonl          # Main data file (append-only)
├── generation_history_YYYY-MM.jsonl  # Monthly archives
├── miner_config_history.jsonl        # Configuration changes log
├── images/                            # Stored input images (IMAGE-TO-3D)
├── ply_files/                        # Validator-accepted PLY files
├── archive/                           # Old compressed archives (>6 months)
└── README.md                          # This file
```

## Data Schema

### generation_history.jsonl

Each line is a JSON object representing one generation:

```json
{
  "metadata": {
    "generation_id": "uuid-v4",
    "timestamp": "2025-11-12T21:30:45.123Z",
    "miner_uid": 226,
    "miner_version": "phase1_phase2_v1.0",
    "validator_uid": 57,
    "validator_hotkey": "5F3sa2...",
    "network": "mainnet"
  },
  "task": {
    "task_type": "IMAGE-TO-3D" | "TEXT-TO-3D",
    "prompt": "text or [BASE64_IMAGE_HASH:sha256:...]",
    "prompt_length": 1234,
    "is_base64_image": true/false,
    "base64_image_hash": "sha256:abc123...",
    "image_file_path": "images/abc123.png"
  },
  "miner_config": {
    "trellis_sparse_steps": 45,
    "trellis_slat_steps": 35,
    "gaussian_threshold": 50000,
    "clip_threshold": 0.10,
    ...
  },
  "generation": {
    "enhanced_prompt": "LLM-enhanced prompt",
    "negative_prompt": "negative prompt (if any)",
    "timing": {
      "total_time": 21.45,
      "sdxl_time": 8.62,
      "background_removal_time": 1.58,
      "trellis_time": 12.15,
      ...
    },
    "output": {
      "num_gaussians": 398234,
      "file_size_mb": 35.3,
      "ply_file_path": "ply_files/uuid_hash.ply",
      "ply_file_hash": "sha256:def456...",
      "quality_metrics": {
        "clip_score": 0.189,
        "clip_threshold_pass": true,
        ...
      },
      "gaussian_stats": {
        "opacity_mean": 4.84,
        ...
      }
    },
    "submission": {
      "submitted": true,
      "submission_time": "2025-11-12T21:31:06Z",
      "pre_submission_rejection": false,
      "rejection_reason": null
    }
  },
  "validator_feedback": {
    "received": true,
    "feedback_time": "2025-11-12T21:32:15Z",
    "score": 0.65,
    "accepted": true,
    "feedback_delay_seconds": 69.2
  },
  "failure_analysis": {
    "failed": false,
    "error_type": null,
    "error_message": null,
    "stack_trace": null
  }
}
```

## Usage

### View Statistics

```bash
# Overall summary
python scripts/analyze_generations.py --summary

# Last 7 days acceptance rate
python scripts/analyze_generations.py --acceptance-rate --last-days 7

# Failure analysis for IMAGE-TO-3D
python scripts/analyze_generations.py --failures --task-type IMAGE-TO-3D

# CLIP score distribution
python scripts/analyze_generations.py --clip-distribution

# Analyze specific validator
python scripts/analyze_generations.py --validator 57
```

### Export Training Data

```bash
# Export high-quality IMAGE-TO-3D generations
python scripts/export_training_data.py \
    --task-type IMAGE-TO-3D \
    --min-score 0.7 \
    --min-clip 0.18 \
    --output data/training/image_to_3d_accepted.jsonl

# Export all accepted generations from last 30 days
python scripts/export_training_data.py \
    --min-score 0.5 \
    --last-days 30 \
    --output data/training/last_month.jsonl
```

### Log Rotation

Logs rotate monthly automatically. To manually trigger:

```bash
bash scripts/rotate_logs.sh
```

## Storage Requirements

### Typical Usage (after 1 month on mainnet)

Assuming 500 generations/day:

- **JSONL logs**: ~1-2 MB/day = 30-60 MB/month
- **Images** (IMAGE-TO-3D, 89% of workload, 512x512 PNG): ~400 KB/image × 445/day = 180 MB/day = 5.4 GB/month
- **PLY files** (accepted only, ~40% acceptance): ~35 MB/file × 200/day = 7 GB/day = 210 GB/month

**Total: ~215 GB/month**

### Storage Management

If storage becomes limited:

1. **Disable image storage**: Set `store_images=False` in data logger initialization
   - Saves ~5.4 GB/month
   - Still stores image hashes for deduplication

2. **Disable PLY storage**: Set `store_ply_files=False`
   - Saves ~210 GB/month
   - Only stores PLY hashes

3. **Compress old archives**: Run rotation script monthly
   - Compresses files >3 months old
   - Moves files >6 months to archive/

4. **Delete old archives**: Manually delete archived data >12 months

## Data Analysis Examples

### Calculate Acceptance Rate

```bash
# Overall acceptance rate
python scripts/analyze_generations.py --acceptance-rate

# By task type
python scripts/analyze_generations.py --acceptance-rate --task-type IMAGE-TO-3D
```

### Identify Best Validators

```bash
# Analyze each validator's acceptance rate
for validator_uid in 27 49 57 128 142 212; do
    echo "=== Validator $validator_uid ==="
    python scripts/analyze_generations.py --validator $validator_uid
done
```

### Track Quality Over Time

```bash
# Compare last 7 days vs previous 7 days
python scripts/analyze_generations.py --summary --last-days 7
python scripts/analyze_generations.py --summary --last-days 14
```

## Fine-Tuning Workflow (Months 3-6)

Once you have 500-1000 validator-accepted examples:

1. **Export high-quality data**:
   ```bash
   python scripts/export_training_data.py \
       --min-score 0.7 \
       --min-clip 0.22 \
       --output training_data.jsonl
   ```

2. **Load images and PLY files** using file paths in exported data

3. **Fine-tune models**:
   - SDXL-Turbo: Improve text-to-image for 3D generation
   - TRELLIS: Improve 3D reconstruction quality
   - BiRefNet: Improve background removal for thin structures

4. **Deploy fine-tuned models** and compare performance

## Backup Strategy

**Recommended**: Sync to cloud storage monthly

```bash
# Example: Sync to S3
aws s3 sync /home/kobe/404-gen/v1/3D-gen/data/ \
    s3://404-gen-miner-data-backup/$(date +%Y-%m)/ \
    --exclude "*.tmp"
```

## Troubleshooting

### Issue: Logs not appearing

**Check**:
1. Data logger initialized? Check miner startup logs
2. Permissions: Ensure `data/` directory is writable
3. Disk space: Check with `df -h`

### Issue: Missing images or PLY files

**Check**:
1. Storage flags: `store_images` and `store_ply_files` enabled?
2. Disk space: May have filled up mid-generation
3. File paths in JSON: Verify relative paths are correct

### Issue: Analysis script errors

**Check**:
1. JSON format: Ensure no corrupted lines in JSONL files
2. Dependencies: Install required packages (`python3 -m pip install -r requirements.txt`)

## Contact

For issues or questions about the data collection system, check:
- Miner logs: `pm2 logs miner-sn17-mainnet`
- Generation service logs: `pm2 logs gen-worker-1`

---

**Last Updated**: 2025-11-12
**Version**: 1.0
**Maintainer**: 404-GEN Team
