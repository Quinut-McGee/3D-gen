# Data Collection System - Integration Guide

**Status**: ✅ Core system implemented and tested
**Next Step**: Integrate into live miner (when ready to deploy)

---

## What's Been Implemented

### Phase 1: Core Data Logger ✅
- **File**: `generation/data_logger.py` (579 lines)
- **Features**:
  - Thread-safe JSONL logging
  - Atomic file writes
  - Image storage (base64 → PNG files)
  - PLY file storage (validator-accepted only)
  - SHA256 hashing for deduplication
  - In-memory pending generations (for validator feedback)
  - Configuration change tracking
  - Statistics API

- **Testing**: All tests passed ✅
  - Single generation logging
  - Concurrent writes (20 parallel generations)
  - Image storage
  - Failure logging
  - Statistics calculation

### Phase 2: Analysis Tools ✅
- **File**: `scripts/analyze_generations.py` (430 lines)
  - Summary statistics
  - Acceptance rate analysis
  - Failure mode analysis
  - CLIP score distribution
  - Validator-specific analysis

- **File**: `scripts/export_training_data.py` (190 lines)
  - Export high-quality data for fine-tuning
  - Filter by score, CLIP, gaussian count
  - Simplified training data format

- **File**: `scripts/rotate_logs.sh` (100 lines)
  - Monthly log rotation
  - Compression (>3 months)
  - Archiving (>6 months)

### Phase 3: Documentation ✅
- **File**: `data/README.md`
  - Complete data schema documentation
  - Usage examples
  - Storage requirements (~215 GB/month)
  - Troubleshooting guide

### Phase 4: Directory Structure ✅
```
/home/kobe/404-gen/v1/3D-gen/data/
├── generation_history.jsonl          # Main log file
├── images/                            # Stored input images
├── ply_files/                        # Validator-accepted outputs
├── archive/                           # Old archives
└── README.md                          # Documentation
```

---

## Integration Instructions

### Step 1: Initialize Data Logger in Generation Service

**File**: `generation/serve_competitive.py`

Add to imports (near line 40):
```python
from data_logger import init_logger, get_logger
```

Add to startup function (after line 670 - where app state is initialized):
```python
# Initialize data logger
init_logger(
    data_dir="/home/kobe/404-gen/v1/3D-gen/data",
    miner_uid=226,
    miner_version="phase1_phase2_v1.1",
    network="mainnet",
    store_images=True,  # Set to False if storage limited
    store_ply_files=True  # Only stores validator-accepted (saves space)
)
app.state.data_logger = get_logger()
logger.info("📊 Data logger initialized")
```

### Step 2: Integrate into Generation Pipeline

**Location**: `generation/serve_competitive.py` - `@app.post("/generate/")` function

#### 2.1: Start Logging (beginning of generation)

Add after line 757 (after detecting task type):
```python
# Start data logging
log_id = None
if hasattr(app.state, 'data_logger') and app.state.data_logger:
    try:
        log_id = app.state.data_logger.start_generation(
            task_type="IMAGE-TO-3D" if is_base64_image else "TEXT-TO-3D",
            prompt=prompt,
            validator_uid=getattr(request, 'validator_uid', 0),  # Get from request if available
            validator_hotkey=getattr(request, 'validator_hotkey', None),
            miner_config={
                "sdxl_turbo_steps": args.flux_steps,
                "trellis_sparse_steps": 45,  # Current config
                "trellis_slat_steps": 35,
                "trellis_sparse_cfg": 9.0,
                "trellis_slat_cfg": 4.0,
                "gaussian_threshold": args.min_gaussian_count,
                "clip_threshold": args.validation_threshold,
                "background_threshold": args.background_threshold,
                "validation_enabled": args.enable_validation,
                "prompt_enhancement_enabled": True,
                "image_enhancement_enabled": args.enable_image_enhancement,
                "depth_estimation_enabled": args.enable_depth_estimation
            }
        )
        logger.debug(f"📝 Started logging generation: {log_id}")
    except Exception as e:
        logger.warning(f"Failed to start data logging: {e}")
```

#### 2.2: Log Enhanced Prompt (after LLM enhancement)

Add after line 890 (after enhanced_prompt is set):
```python
# Log enhanced prompt
if log_id and hasattr(app.state, 'data_logger'):
    try:
        app.state.data_logger.log_enhanced_prompt(
            log_id,
            enhanced_prompt=enhanced_prompt,
            negative_prompt=tier1_negative_prompt if 'tier1_negative_prompt' in locals() else None
        )
    except Exception as e:
        logger.debug(f"Failed to log enhanced prompt: {e}")
```

#### 2.3: Log Timing (after each component)

Add after SDXL timing (around line 922):
```python
if log_id:
    app.state.data_logger.log_timing(log_id, "sdxl_time", t2 - t1)
```

Add after background removal (around line 945):
```python
if log_id:
    app.state.data_logger.log_timing(log_id, "background_removal_time", t3 - t2)
```

Add after depth estimation (around line 985):
```python
if log_id:
    app.state.data_logger.log_timing(log_id, "depth_estimation_time", time.time() - t2_5_start)
```

Add after TRELLIS (around line 1022):
```python
if log_id:
    app.state.data_logger.log_timing(log_id, "trellis_time", t4 - t3_start)
```

Add after validation (around line 1170):
```python
if log_id:
    app.state.data_logger.log_timing(log_id, "validation_time", time.time() - t_validation_start)
```

#### 2.4: Log Output (after generation complete)

Add after line 1175 (after all outputs are available):
```python
# Log generation output
if log_id and hasattr(app.state, 'data_logger'):
    try:
        app.state.data_logger.log_output(
            log_id,
            num_gaussians=timings.get('num_gaussians'),
            file_size_mb=timings.get('file_size_mb'),
            clip_score=three_d_clip_score,
            clip_threshold_pass=three_d_clip_score >= args.validation_threshold if three_d_clip_score else None,
            gaussian_count_pass=timings.get('num_gaussians', 0) >= args.min_gaussian_count if timings.get('num_gaussians') else None,
            validation_pass=validation_passed,
            gaussian_stats={
                "opacity_mean": timings.get('opacity_mean'),
                "opacity_std": timings.get('opacity_std'),
                # Add more stats if available from gs_model
            } if timings.get('opacity_mean') is not None else None,
            render_stats={
                "num_views_rendered": 4 if three_d_clip_score else 0,
                "render_clip_scores": [three_d_clip_score] * 4 if three_d_clip_score else []
            } if three_d_clip_score else None
        )
    except Exception as e:
        logger.warning(f"Failed to log output: {e}")
```

#### 2.5: Log Submission Status

Add before returning response (around line 1280):
```python
# Log submission status
if log_id and hasattr(app.state, 'data_logger'):
    try:
        if validation_passed:
            app.state.data_logger.log_submission(log_id, submitted=True)
            # Note: Validator feedback will be logged later when received
        else:
            app.state.data_logger.log_submission(
                log_id,
                submitted=False,
                rejection_reason="validation_failed"
            )
            app.state.data_logger.finalize_generation(log_id)
    except Exception as e:
        logger.warning(f"Failed to log submission: {e}")
```

#### 2.6: Log Failures (in exception handler)

Add in exception handlers (around line 1281):
```python
except Exception as e:
    # Existing error handling...

    # Log failure
    if log_id and hasattr(app.state, 'data_logger'):
        try:
            error_type = "crash"
            if "timeout" in str(e).lower():
                error_type = "timeout"
            elif "memory" in str(e).lower() or "oom" in str(e).lower():
                error_type = "oom"

            app.state.data_logger.log_failure(
                log_id,
                error_type=error_type,
                error_message=str(e),
                stack_trace=traceback.format_exc()
            )
            app.state.data_logger.finalize_generation(log_id)
        except Exception as log_err:
            logger.warning(f"Failed to log failure: {log_err}")

    # Re-raise or return error response...
```

### Step 3: Log Validator Feedback

**File**: `neurons/miner/competitive_miner.py` or wherever validator feedback is received

When you receive validator score:
```python
# After receiving validator feedback
if hasattr(app.state, 'data_logger') and log_id:
    try:
        app.state.data_logger.log_validator_feedback(
            log_id,
            score=validator_score
        )
        app.state.data_logger.finalize_generation(log_id)
    except Exception as e:
        logger.warning(f"Failed to log validator feedback: {e}")
```

**Note**: You'll need to pass `log_id` from generation service to miner. Options:
1. Store in app state with generation timestamp as key
2. Return in response headers
3. Use a shared dict/cache keyed by prompt hash

### Step 4: Deploy and Verify

```bash
# 1. Restart generation service
pm2 restart gen-worker-1

# 2. Check logs for initialization
pm2 logs gen-worker-1 | grep "Data logger"

# 3. Generate a test object (TEXT-TO-3D or IMAGE-TO-3D)
curl -X POST http://localhost:10010/generate/ -d "prompt=test object"

# 4. Verify data logged
tail -1 /home/kobe/404-gen/v1/3D-gen/data/generation_history.jsonl | python3 -m json.tool

# 5. Check statistics
python3 scripts/analyze_generations.py --summary
```

---

## Storage Considerations

### Current Storage Available
Check with:
```bash
df -h /home/kobe/404-gen/v1/3D-gen/data
```

### Expected Usage (500 generations/day)
- **JSONL logs**: 1-2 MB/day = 30-60 MB/month
- **Images**: 180 MB/day = 5.4 GB/month
- **PLY files** (accepted): 7 GB/day = 210 GB/month
- **Total**: ~215 GB/month

### If Storage Limited

Option 1: Disable PLY storage (saves ~210 GB/month)
```python
init_logger(
    store_images=True,
    store_ply_files=False  # Only store hashes, not files
)
```

Option 2: Disable image storage (saves ~5.4 GB/month)
```python
init_logger(
    store_images=False,  # Only store hashes
    store_ply_files=True
)
```

Option 3: Setup automatic backup and deletion
```bash
# Monthly cron: backup to cloud and delete local
0 0 1 * * bash /home/kobe/404-gen/v1/3D-gen/scripts/backup_and_clean.sh
```

---

## Monitoring After Deployment

### Daily Checks

```bash
# Check data collection is working
python3 scripts/analyze_generations.py --summary --last-days 1

# Expected output after 1 day:
# Total generations: ~500
# Acceptance rate: 40-50%
# Average CLIP: 0.18-0.26
```

### Weekly Analysis

```bash
# Acceptance rate trend
python3 scripts/analyze_generations.py --acceptance-rate --last-days 7

# Failure analysis
python3 scripts/analyze_generations.py --failures

# Validator comparison
for uid in 27 49 57 128 142 212; do
    echo "=== Validator $uid ==="
    python3 scripts/analyze_generations.py --validator $uid
done
```

### Monthly Tasks

```bash
# 1. Rotate logs
bash scripts/rotate_logs.sh

# 2. Export training data (if enough accepted)
python3 scripts/export_training_data.py \
    --min-score 0.7 \
    --output data/training/$(date +%Y-%m)_accepted.jsonl

# 3. Check storage usage
du -sh data/
```

---

## Troubleshooting

### Issue: Data not being logged

**Check**:
```bash
# 1. Verify data logger initialized
pm2 logs gen-worker-1 | grep "Data logger initialized"

# 2. Check for errors
pm2 logs gen-worker-1 --err | grep -i "data.*log"

# 3. Verify directory permissions
ls -la /home/kobe/404-gen/v1/3D-gen/data/

# 4. Check disk space
df -h /home/kobe/404-gen
```

### Issue: Missing validator feedback

**Check**:
```bash
# Validator feedback requires passing log_id from generation to miner
# Check if log_id is being passed correctly
grep "log_validator_feedback" ~/.pm2/logs/*.log
```

**Solution**: Implement one of the log_id passing mechanisms (see Step 3)

### Issue: High disk usage

**Check**:
```bash
# Breakdown by directory
du -sh data/*/

# Most common:
# - data/ply_files/ (largest)
# - data/images/ (medium)
# - data/*.jsonl (smallest)
```

**Solution**:
- Disable PLY storage if needed
- Run rotation script monthly
- Setup cloud backup and local cleanup

---

## Next Steps

### Immediate (when ready to deploy)
1. Review this guide
2. Integrate data logger into serve_competitive.py (follow Step 2)
3. Test with a few generations
4. Deploy to mainnet

### Month 1-2: Monitoring
1. Watch acceptance rates daily
2. Analyze failure patterns
3. Identify best validators

### Month 3-6: Fine-Tuning Preparation
1. Accumulate 500-1000 validator-accepted examples
2. Export high-quality training data
3. Begin model fine-tuning experiments

### Month 6+: Continuous Improvement
1. Fine-tune TRELLIS on validator-preferred outputs
2. Fine-tune SDXL-Turbo on high-scoring prompts
3. Analyze validator preferences and optimize

---

## Files Created

```
generation/
├── data_logger.py                    # Core data logger (579 lines) ✅

scripts/
├── analyze_generations.py            # Analysis tool (430 lines) ✅
├── export_training_data.py           # Training data export (190 lines) ✅
└── rotate_logs.sh                    # Log rotation (100 lines) ✅

data/
├── README.md                          # Data documentation ✅
├── generation_history.jsonl          # Created on first generation
├── images/                            # Created ✅
├── ply_files/                        # Created ✅
└── archive/                           # Created ✅

/tmp/
└── test_data_logger.py               # Test suite (passed) ✅
```

---

## Summary

**What's Done**: ✅
- Core data logger implemented and tested
- Analysis and export tools ready
- Documentation complete
- Directory structure created

**What's Next**:
- Integrate into serve_competitive.py (when ready)
- Deploy and verify
- Start collecting mainnet data

**Benefits**:
- Track all generations for analysis
- Identify failure patterns
- Collect training data for fine-tuning
- Monitor miner performance over time
- Build competitive advantage with data-driven optimization

---

**Ready to integrate when you're ready to deploy!**
