# Deployment Checklist for 404-GEN Miner Optimizations

## Files Modified

### 1. Bug Fixes
- [x] `generation/DreamGaussianLib/GaussianProcessor.py` - Fixed typo (pemute → permute)

### 2. Diagnostics
- [x] `neurons/miner/validator_selector.py` - Added detailed validator selection logging

### 3. Quality Improvements
- [x] `generation/configs/text_mv.yaml` - Updated training parameters
  - iters: 1000 → 1500
  - num_pts: 5000 → 8000
  - guidance_scale: 120 (new)
  - density_end_iter: 3000 → 1200
  - position_lr_max_steps: 300 → 450
  - Enhanced negative_prompt

- [x] `generation/DreamGaussianLib/GaussianProcessor.py` - Pass guidance_scale to models
- [x] `generation/DreamGaussianLib/AIModelsUtils/mvdream_utils.py` - Support configurable guidance_scale
- [x] `generation/DreamGaussianLib/AIModelsUtils/sd_utils.py` - Support configurable guidance_scale
- [x] `generation/DreamGaussianLib/AIModelsUtils/imagedream_utils.py` - Support configurable guidance_scale

### 4. Quality Assurance
- [x] `neurons/miner/workers.py` - Added pre-submission size validation

### 5. Documentation
- [x] `OPTIMIZATION_CHANGES.md` - Comprehensive change documentation
- [x] `DEPLOYMENT_CHECKLIST.md` - This file

## Pre-Deployment Checks

Before restarting your miner, verify:

### 1. Check Your Miner Config
```bash
# Find your config file (usually in ~/.bittensor or similar)
cat neurons/miner/config.py
```

Look for `min_stake_to_set_weights` - if it's high and you're getting "No available validators", consider lowering it.

### 2. Verify GPU Resources
```bash
nvidia-smi
```
- Ensure you have ~20-22GB VRAM available
- Check GPU utilization is low before restart

### 3. Backup Current State
```bash
# Save current PM2 state
pm2 save

# Note current git state
git status
git log -1
```

## Deployment Steps

### Step 1: Restart Generation Service
```bash
pm2 restart generation
pm2 logs generation --lines 50
```

**Expected output**:
- "[INFO] loading MVDream..."
- "[INFO] loaded MVDream!"
- Server starting on port 10006

### Step 2: Test Generation Service
```bash
# In a new terminal
curl -X POST http://localhost:10006/generate/ \
  -F "prompt=a red sports car" \
  --output test_output.ply

# Check file size
ls -lh test_output.ply
```

**Expected**: File should be >50KB, generation takes ~3-4 minutes

### Step 3: Restart Miner
```bash
pm2 restart miner
pm2 logs miner --lines 100
```

**Expected output**:
- "Starting neuron on subnet 17 with uid XXX"
- "Querying task from [X]" (instead of "No available validators")
- "Task received. Prompt: ..."
- "Feedback received from [X]. Score: ..."

### Step 4: Monitor First Few Generations

Watch for these key indicators:

#### Good Signs ✅
- "Querying task from [validator_uid]"
- "Generation passed size check: XXXXX bytes"
- "Feedback received... Score: 0.75" or higher
- "Current miner reward: 0.00X" (increasing)

#### Warning Signs ⚠️
- "No available validators to pull the task" (check min_stake config)
- "Generation too small (XXX bytes)" (check GPU/generation service)
- "Score: failed" or "Score: 0.0" (quality issues)
- Errors/crashes in generation logs

## Monitoring & Validation

### First Hour
Check every 10-15 minutes:
```bash
pm2 logs miner --lines 20
```

Look for:
- Tasks being received
- Successful submissions
- Feedback scores

### First 4 Hours
Check hourly:
- Average CLIP scores improving
- Number of successful generations
- Reward accumulation

### First 24 Hours
Compare metrics:
- Dashboard ELO rating trend
- Total rewards vs previous 24h
- Duel win rate

## Performance Expectations

### Generation Timing
- **Before**: ~2.5 minutes per generation
- **After**: ~3.5-4.0 minutes per generation
- **Throughput drop**: ~35% fewer generations per 8h window

### Quality Metrics
- **Target CLIP score**: ≥0.75 (good), ≥0.80 (excellent)
- **Expected improvement**: +15-25% vs baseline
- **Reward increase**: +40-60% despite lower throughput

### Resource Usage
- **VRAM**: ~20-22GB (up from ~18GB)
- **GPU Utilization**: Should stay 95-100% during generation
- **Generation time**: 3.5-4 min is optimal for quality/speed balance

## Troubleshooting

### Issue: "No available validators"
**Solution**:
1. Check logs for specific skip reasons
2. Lower `min_stake_to_set_weights` in config
3. Verify network connectivity

### Issue: CUDA Out of Memory
**Solutions**:
- Reduce `num_pts` to 6000 in text_mv.yaml
- Reduce `iters` to 1200
- Check no other GPU processes running

### Issue: Generation taking >6 minutes
**Solutions**:
- Check GPU utilization with `nvidia-smi`
- Verify no thermal throttling
- Consider reducing to iters: 1200

### Issue: Low CLIP scores (<0.6)
**Solutions**:
- Increase guidance_scale to 150
- Check negative prompts are being used
- Verify MVDream model loaded correctly

### Issue: PM2 process keeps crashing
**Solutions**:
```bash
# Check detailed error logs
pm2 logs generation --err --lines 100

# Check Python environment
which python
python --version

# Restart with manual logging
pm2 delete generation
cd generation && python serve.py --port 10006
```

## Rollback Procedure

If you need to revert all changes:

```bash
# Stop services
pm2 stop all

# Revert code changes
git checkout HEAD -- generation/configs/text_mv.yaml
git checkout HEAD -- generation/DreamGaussianLib/GaussianProcessor.py
git checkout HEAD -- generation/DreamGaussianLib/AIModelsUtils/mvdream_utils.py
git checkout HEAD -- generation/DreamGaussianLib/AIModelsUtils/sd_utils.py
git checkout HEAD -- generation/DreamGaussianLib/AIModelsUtils/imagedream_utils.py
git checkout HEAD -- neurons/miner/validator_selector.py
git checkout HEAD -- neurons/miner/workers.py

# Restart services
pm2 restart all
```

## Success Criteria

After 24 hours, you should see:

✅ **Minimum Success**:
- Miner receiving and completing tasks
- Average CLIP scores ≥0.70
- Positive reward accumulation
- No crashes or errors

✅ **Good Success**:
- Average CLIP scores ≥0.75
- Rewards +20% vs previous 24h
- Winning some ELO duels
- Stable operation

✅ **Excellent Success**:
- Average CLIP scores ≥0.80
- Rewards +40-60% vs previous 24h
- ELO rating improving
- Winning majority of duels

## Next Optimization Phase

Once stable at this level, consider:
1. Adding local CLIP pre-validation
2. Running multiple generation endpoints
3. Prompt preprocessing/enhancement
4. Category-specific parameter tuning
5. Advanced quality filtering

## Support

If you encounter issues:
1. Check logs first: `pm2 logs miner --err --lines 100`
2. Verify GPU status: `nvidia-smi`
3. Review this checklist for known issues
4. Check subnet Discord/GitHub for similar issues
5. Review OPTIMIZATION_CHANGES.md for detailed explanations

---

**Deployment Date**: _____________
**Initial Metrics**: ELO: _____ | Avg CLIP: _____ | Reward/8h: _____
**Post-Deploy Metrics** (24h): ELO: _____ | Avg CLIP: _____ | Reward/8h: _____
