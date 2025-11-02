# CLIP Validation Threshold Fix - URGENT

## Problem Summary

Your miner is rejecting 50%+ of valid submissions due to an overly aggressive CLIP validation threshold.

**Evidence:**
- Validator rejection threshold: raw CLIP < 0.105 (normalized 0.3)
- Your rejection threshold: 0.20-0.25 (2-3x too high!)
- Result: Rejecting CLIP scores of 0.173-0.216 that validators would score 0.494-0.617 (well above their 0.3 threshold)

**Impact:**
- Low submission rate → inadequate validator feedback → ELO rating drops → kicked from mainnet

## The Fix (3 Options)

### Option 1: DISABLE CLIP Validation Entirely (RECOMMENDED FOR NOW)

**Rationale:**
- TRELLIS outputs are already high quality (0.682-0.887 validator scores when submitted)
- CLIP validation is causing more harm than good by rejecting valid submissions
- Validators already have their own quality checks

**How to implement:**
```bash
# Stop the worker
pm2 stop gen-worker-1

# Edit PM2 ecosystem or start command to REMOVE --enable-validation flag
# Current: --port 10010 --enable-validation --validation-threshold 0.XX
# Fixed:   --port 10010
# (no --enable-validation flag)

# Restart
pm2 restart gen-worker-1
pm2 save
```

### Option 2: Lower Threshold to Match Validator Reality (SAFER)

**Rationale:**
- Keep validation but align with validator thresholds
- Safety margin above actual validator rejection point

**How to implement:**
```bash
# Validator rejects at raw CLIP < 0.105
# Add 30% safety margin: 0.105 × 0.7 = 0.074
# Round up for safety: 0.10

pm2 stop gen-worker-1

# Update start command:
--enable-validation --validation-threshold 0.10

pm2 restart gen-worker-1
pm2 save
```

### Option 3: Adaptive Threshold Based on Submission Rate (ADVANCED)

Monitor submission rate and dynamically adjust threshold:
- If submission rate < 80% of task rate → lower threshold by 0.02
- If rejection rate > 50% → disable validation temporarily
- Requires code changes (not recommended for immediate fix)

## Recommended Immediate Action

**Step 1: Disable CLIP validation entirely**
```bash
cd /home/kobe/404-gen/v1/3D-gen

# Check current PM2 config
pm2 describe gen-worker-1 | grep "script args"

# Stop worker
pm2 stop gen-worker-1

# Start without validation
pm2 start generation/serve_competitive.py \\
  --name gen-worker-1 \\
  --interpreter /home/kobe/miniconda3/envs/three-gen-mining/bin/python \\
  -- --port 10010

pm2 save
```

**Step 2: Restart TRELLIS microservice**
```bash
pm2 start trellis-microservice
pm2 save

# Verify it's healthy
sleep 10
curl http://localhost:10008/health
```

**Step 3: Restart miner**
```bash
pm2 restart miner-sn17-mainnet

# Monitor for 5 minutes
pm2 logs miner-sn17-mainnet --lines 100
```

**Step 4: Verify submissions are flowing**
```bash
# Watch for successful submissions (should see compress times > 0.05s)
pm2 logs miner-sn17-mainnet | grep "Submission to"

# Should see patterns like:
# compress=0.11s, compress=0.16s, etc. (NOT compress=0.00s)
```

## Expected Results

**Before fix:**
- Submission rate: 50% of tasks (half rejected by CLIP validation)
- Validator scores: 0.682-0.887 (when submissions got through)
- Problem: Too few submissions → poor Glicko2 rating

**After fix:**
- Submission rate: 95%+ of tasks (only truly bad outputs rejected)
- Validator scores: Expected to maintain 0.60-0.85 range
- Result: Consistent submissions → stable Glicko2 rating → stay on mainnet

## Monitoring Checklist

After implementing the fix, monitor for 1 hour:

- [ ] Worker generating successfully (check logs for "✅ Generation successful")
- [ ] No CLIP validation rejections (or very few if using Option 2)
- [ ] Submissions showing compress times > 0.00s
- [ ] Validator feedback showing scores > 0.0 (at least 70% should be non-zero)
- [ ] Glicko2 rating stable or increasing

## Rollback Plan

If fix causes issues:

```bash
# Stop everything
pm2 stop all

# Revert to previous configuration
# (Check git for previous PM2 startup script)

pm2 start all
```

## Long-term Solution

Once mainnet is stable:

1. Collect 1000+ submissions with validator feedback
2. Run diagnostics/correlate_validator_feedback.py to find optimal threshold
3. Re-enable validation with data-driven threshold (likely ~0.10-0.12)
4. Monitor and adjust based on actual rejection rates

## Key Insight

**The validator normalization factor (0.35) means raw CLIP scores of 0.15-0.35 are NORMAL and ACCEPTABLE.**

Your threshold of 0.20-0.25 was rejecting the majority of this normal range, leaving only the top 10-20% of outputs. This is like throwing away 80% of your inventory when 90% of it would sell!

## Questions?

If you're unsure about any step, ask before proceeding. But the safest immediate action is:
**Disable CLIP validation entirely and let validators do their job.**
