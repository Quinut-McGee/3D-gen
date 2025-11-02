# TRELLIS Enhancement Fix - Nov 2, 2025

## Problem Summary

TRELLIS was generating sparse outputs (57K-97K gaussians) 50%+ of the time, causing validator rejections (Score=0.0) and mainnet ejection.

## Root Cause

1. **Missing Quality Gate** (during initial deploy): Sparse outputs were submitted without filtering → Score=0.0
2. **Insufficient Image Enhancement**: Even with quality gate, TRELLIS produces too many sparse outputs
3. **No Retry Logic**: One-shot generation means 50% failure rate

## Implemented Fixes

### Fix 1: Increased Image Enhancement Aggressiveness

**Before:**
- Sharpness: 2.5x
- Contrast: 1.5x

**After:**
- Sharpness: 3.5x (+40% increase)
- Contrast: 1.8x (+20% increase)

This gives TRELLIS more surface features to detect → denser voxel grids → more gaussians.

### Fix 2: Added Retry Logic with Extreme Enhancement

**New Behavior:**
1. First attempt: Standard enhancement (3.5x sharpness, 1.8x contrast)
2. If sparse (<150K gaussians): Retry with extreme enhancement
   - Sharpness: 5.0x
   - Contrast: 2.5x
   - Edge enhancement filter
   - Brightness boost: 1.1x
3. Use better result (even if still slightly sparse)

**Expected Impact:**
- Borderline cases (100K-149K) should now pass 150K threshold
- Success rate should improve from 50% → 70-80%

## Code Changes

File: `generation/trellis_integration.py`

### Changed Functions:
1. `enhance_image_for_trellis()` - Increased enhancement values
2. `generate_with_trellis()` - Added retry logic
3. **NEW:** `apply_extreme_enhancement()` - Maximum enhancement for retries
4. **NEW:** `_call_trellis_api()` - Helper for API calls

### Retry Logic Flow:

```python
# Attempt 1: Standard enhancement
result = generate_with_trellis_enhanced(image)

if result.gaussians < 150K:
    # Attempt 2: Extreme enhancement
    retry_result = generate_with_extreme_enhancement(image)

    if retry_result.gaussians >= 150K:
        return retry_result  # ✅ Retry succeeded
    elif retry_result.gaussians > result.gaussians:
        return retry_result  # ⚠️  Better than first attempt
    else:
        raise ValueError  # ❌ Both attempts failed
```

## Monitoring Commands

### Check Success Rate:
```bash
cd /home/kobe/404-gen/v1/3D-gen

# Count sparse rejections (failures)
grep "SPARSE GENERATION\|❌ QUALITY GATE FAILED" ~/.pm2/logs/gen-worker-1-error.log | tail -100 | wc -l

# Count successful generations
grep "✅ TRELLIS generation done" ~/.pm2/logs/gen-worker-1-error.log | tail -100 | wc -l

# Count retry attempts
grep "🔄 Retry result\|⚠️  SPARSE GENERATION" ~/.pm2/logs/gen-worker-1-error.log | tail -50

# Count successful retries
grep "✅ RETRY SUCCESSFUL" ~/.pm2/logs/gen-worker-1-error.log | tail -20
```

### Watch Live Activity:
```bash
# Watch for retry attempts
pm2 logs gen-worker-1 | grep -E "SPARSE|Retry|RETRY"

# Monitor validator feedback
pm2 logs miner-sn17-mainnet | grep "Feedback from"
```

## Expected Results

### Before Fix:
- Success rate: ~50% (50/100 generations pass quality gate)
- Validator rejections: High (sparse outputs get Score=0.0)
- Mainnet status: Kicked after 24 hours

### After Fix:
- Success rate: ~70-80% (70-80/100 generations pass quality gate)
- Validator rejections: Low (mostly dense outputs, Score 0.6-0.8)
- Mainnet status: Stable

### Retry Statistics (Expected):
- Retry attempts: ~30-40% of generations
- Retry success rate: ~50-60% (half of retries should succeed)
- Overall improvement: 15-20 percentage points

## Validation Checklist

Before returning to mainnet, verify:

- [ ] Success rate > 70% over 100 generations
- [ ] Retry logic is working (see "🔄 Retry result" in logs)
- [ ] Successful retries are happening (see "✅ RETRY SUCCESSFUL")
- [ ] No excessive errors or crashes
- [ ] Validator feedback shows improvement (fewer Score=0.0)

## Rollback Plan

If fix causes issues:

```bash
# Revert to previous version
cd /home/kobe/404-gen/v1/3D-gen
git checkout HEAD~1 generation/trellis_integration.py

# Restart services
pm2 restart gen-worker-1
```

## Next Steps

1. **Monitor for 1 hour**: Watch logs, check success rate
2. **Test on testnet**: Register on testnet (netuid 220), run for 2 hours
3. **Validate metrics**: Confirm > 70% success rate, validator scores improving
4. **Return to mainnet**: If testnet successful, re-register on mainnet
5. **Long-term tuning**: May need to adjust enhancement values based on data

## Key Insight

TRELLIS quality is excellent when it generates dense outputs (0.682-0.887 validator scores). The problem is frequency of sparse outputs, not the quality of dense ones. The fix focuses on:
1. Making more outputs dense (increased enhancement)
2. Recovering borderline cases (retry logic)
3. Maintaining the excellent quality of successful outputs

Success = Consistency, not peak quality.
