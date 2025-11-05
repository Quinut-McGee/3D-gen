# Phase 3: Deep Dive Results & Path Forward

**Analysis Date:** 2025-11-04 21:42 UTC
**Data Period:** 21:00-21:41 (41 minutes, 21 submissions)
**Success Rate:** 42.9% (9/21)

---

## Executive Summary

**Phase 3 Status: ⚠️ PARTIAL SUCCESS**

✅ **What Worked:**
- Success rate improved from 38.5% (Phase 2) to 42.9% (+4.4 percentage points)
- Back to baseline performance (~42%)
- CFG increase showed modest positive impact
- High-density outputs (700K-800K gaussians) ARE being accepted

🚨 **Critical Discovery:**
- **14.3% of generations (3/21) have complete opacity corruption**
- These are guaranteed rejections
- Fixing this bug alone would add +2-3% success rate

📊 **Performance vs Targets:**
- Current: 42.9%
- Phase 3 Target: 55%+
- Shortfall: -12 percentage points below target

---

## Critical Finding: The Opacity Corruption Bug

### The Smoking Gun

**3 generations had complete opacity corruption:**

| Time | Validator | Gaussians | Avg Opacity | Opacity Std | Result |
|------|-----------|-----------|-------------|-------------|--------|
| 21:01:03 | 128 | 489,088 | **-9.21** | **0.00** | Score 0.0 ❌ |
| 21:34:40 | 212 | 762,112 | **-9.16** | **0.76** | Score 0.0 ❌ |
| 21:37:16 | 27 | 400,320 | **-9.21** | **0.00** | Score 0.0 ❌ |

### What This Means

**Normal Healthy Generation:**
- avg_opacity: 2-8 (logit space, indicates visible gaussians)
- opacity_std: 4-9 (healthy variation between opaque/transparent)

**Corrupted Generation:**
- avg_opacity: -9.21 (ALL gaussians invisible)
- opacity_std: 0.0 (NO variation, all identical)
- Result: Validators see completely invisible/empty model

### Impact

- **3/12 failures (25%) caused by this bug**
- These are 100% guaranteed rejections
- Validators cannot score invisible models
- This is a systematic issue, not random variance

### Root Cause

TRELLIS occasionally outputs corrupted opacity values where:
1. All gaussians get clamped to minimum opacity (-9.21 in logit space)
2. No variation in opacity (std dev = 0)
3. This may be related to specific prompts, seeds, or edge cases
4. The corruption happens during TRELLIS generation, not post-processing

---

## Performance Analysis

### Success Rate Progression

| Phase | Success Rate | Change | Key Action |
|-------|--------------|--------|------------|
| Baseline (Pre-Phase 1) | 42.4% | - | - |
| Phase 1 (Blacklist) | ~45% | +2.6% | Blacklisted V81, V199 |
| Phase 2 (Quality Gate) | 38.5% | -6.5% ❌ | Added 150K threshold |
| **Phase 3 (CFG Increase)** | **42.9%** | **+4.4%** | **Increased CFG strength** |

**Conclusion:** Phase 3 recovered from Phase 2's decline, returning to baseline.

### Validator Performance Breakdown

| Validator | Success Rate | Successes | Failures | Notes |
|-----------|--------------|-----------|----------|-------|
| **212** | **75.0%** | 3/4 | 1 | ✅ Excellent - most reliable |
| 49 | 50.0% | 3/6 | 3 | ⚠️ Moderate |
| 128 | 40.0% | 2/5 | 3 | ⚠️ Below average |
| 27 | 25.0% | 1/4 | 3 | ❌ Very strict |
| **142** | **0.0%** | 0/2 | 2 | ❌ **Never accepts** |

**Key Findings:**
- Validator 142: 0% acceptance (impossible standards or broken)
- Validator 27: 25% acceptance (very strict, 1 of 3 failures was opacity corruption)
- Validator 212: 75% acceptance (best performer, but 1 failure was opacity corruption)

### Success Score Distribution

**Successful Scores (n=9):**
- Average: 0.729
- Median: 0.714
- Range: 0.666 - 0.860

**Top 3 Scores:**
1. 0.860 - Validator 49 (309K gaussians)
2. 0.805 - Validator 27 (803K gaussians) ⭐
3. 0.732 - Validator 49 (551K gaussians)

**Insight:** When validators accept, they give high scores (0.66-0.86). The issue is rejection rate, not score quality.

---

## Metric Comparison: Successes vs Failures

| Metric | Successes (Avg) | Failures (Avg) | Difference |
|--------|----------------|----------------|------------|
| **Gaussians** | 499,527 | 444,259 | -55K (successes have MORE) |
| **Avg Opacity** | 4.64 | -2.57 | +7.2 ⚠️ **Failures NEGATIVE** |
| **Opacity Std** | 6.84 | 4.21 | +2.6 (successes have healthy variation) |
| **BBox Volume** | 0.450 | 0.524 | +0.07 (failures slightly larger) |
| **Density Max** | 3,137 | 2,828 | -309 (similar) |

### Critical Insights

1. **Successes have MORE gaussians than failures** (counterintuitive!)
2. **Failures have NEGATIVE average opacity** (corruption dragging down average)
3. **Successes have HIGHER opacity std** (healthier variation)
4. **Gaussian density does NOT predict success** (failures range 152K-762K)

**Conclusion:** Opacity health (positive average, high variation) correlates with success more than gaussian count.

---

## What's Working Well

### ✅ High-Density Acceptance

**Proof that high-quality outputs CAN succeed:**
- 803K gaussians → Validator 27 → Score 0.805 ✅
- 735K gaussians → Validator 128 → Score 0.666 ✅
- 720K gaussians → Validator 128 → Score 0.721 ✅

This proves:
- Quality gate not the limiting factor
- High-density outputs ARE being accepted
- The issue is consistency, not capability

### ✅ Validator 212 Reliability

**Best performing validator:**
- 75% success rate (3/4)
- Accepts wide range: 155K-412K gaussians
- Even accepts low-density outputs if quality is good
- One failure was opacity corruption (would've been 100% without bug)

### ✅ CFG Increase Impact

**Phase 3 showed measurable improvement:**
- Phase 2: 38.5% → Phase 3: 42.9% (+4.4%)
- Modest but consistent positive trend
- Suggests prompt adherence does matter
- Not as impactful as hoped, but helpful

---

## Critical Issues to Fix

### 🚨 ISSUE #1: Opacity Corruption (CRITICAL)

**Severity:** CRITICAL - Guaranteed rejections
**Frequency:** 14.3% of generations (3/21)
**Impact:** 25% of failures caused by this bug

**Problem:**
- TRELLIS occasionally outputs opacity_std = 0
- All gaussians become invisible (avg_opacity = -9.21)
- Validators reject invisible models immediately

**Solution:**
```python
# Add validation in trellis_integration.py after generation
if ply_metrics['opacity_std'] < 1.0:
    logger.error(f"❌ OPACITY CORRUPTION: opacity_std = {ply_metrics['opacity_std']}")
    # Option A: Skip submission (safer)
    return None
    # Option B: Regenerate with different seed
    # regenerate_with_seed(seed + 1)
```

**Expected Impact:** +2-3% success rate (eliminate 25% of failures)

### 🚨 ISSUE #2: Validator 27 & 142 Strict Standards

**Severity:** HIGH - Wasting compute on impossible validators
**Impact:** 6 submissions, 1 success (16.7% success rate combined)

**Problem:**
- Validator 27: 25% success (very strict)
- Validator 142: 0% success (never accepts anything)
- Combined: Wasting 28% of submissions on low-yield validators

**Solution:**
```python
# Update validator_selector.py blacklist
BLACKLISTED_VALIDATORS = [
    180,  # Discord FAQ WC
    81,   # Phase 1: 41.2% success
    199,  # Phase 1: 41.5% success
    27,   # Phase 3: 25% success (very strict)
    142,  # Phase 3: 0% success (impossible)
]
```

**Expected Impact:** +3-5% success rate (avoid wasted compute, focus on good validators)

### ⚠️ ISSUE #3: Low-Density Failures

**Severity:** MODERATE - Quality gate may be too permissive
**Impact:** 3 failures with <200K gaussians

**Problem:**
- 152K, 180K, 206K gaussian outputs all rejected
- Current quality gate: 150K (too low)
- Below 200K has poor success rate

**Solution:**
```python
# Update serve_competitive.py
parser.add_argument(
    "--min-gaussian-count",
    type=int,
    default=200000,  # Increased from 150K
    help="Minimum gaussian count threshold"
)
```

**Expected Impact:** +2-4% success rate (filter marginal outputs)

### ⚠️ ISSUE #4: Negative Average Opacity

**Severity:** MODERATE - Indicator of poor visual quality
**Impact:** Multiple failures have avg_opacity < 0

**Problem:**
- Failures average: -2.57 (many invisible gaussians)
- Successes average: +4.64 (healthy opacity)
- Indicates poor opacity distribution

**Solution:**
```python
# Add to quality gate checks
if ply_metrics['avg_opacity'] < 0:
    logger.warning(f"⚠️  Negative avg_opacity: {ply_metrics['avg_opacity']}")
    # Consider skipping or flagging
```

**Expected Impact:** +1-2% success rate (filter poor opacity distributions)

---

## Recommended Actions: Phase 4

### Implementation Priority

**PHASE 4A: Opacity Validation (IMMEDIATE)**

```bash
# 1. Edit trellis_integration.py
# Add after PLY metrics calculation (line ~320):

if ply_quality['opacity_std'] < 1.0:
    logger.error(f"❌ OPACITY CORRUPTION DETECTED!")
    logger.error(f"   opacity_std: {ply_quality['opacity_std']} (should be >1.0)")
    logger.error(f"   avg_opacity: {ply_quality['avg_opacity']}")
    logger.error(f"   Skipping submission to avoid guaranteed rejection")
    return None  # Skip this generation

# 2. Restart gen-worker-1
pm2 restart gen-worker-1 && pm2 save
```

**Expected Impact:** +2-3% success rate
**Risk:** Very low (just filtering bad outputs)
**Time:** 10 minutes

---

**PHASE 4B: Blacklist Validators 27 & 142 (HIGH PRIORITY)**

```bash
# 1. Edit validator_selector.py
# Update BLACKLISTED_VALIDATORS to:
BLACKLISTED_VALIDATORS = [180, 81, 199, 27, 142]

# 2. Restart miner
pm2 restart miner-sn17-mainnet && pm2 save
```

**Expected Impact:** +3-5% success rate
**Risk:** Low (avoiding low-yield validators)
**Time:** 5 minutes

---

**PHASE 4C: Increase Quality Gate to 200K (MEDIUM PRIORITY)**

```bash
# 1. Edit serve_competitive.py
# Change default=150000 to default=200000

# 2. Restart gen-worker-1
pm2 restart gen-worker-1 && pm2 save
```

**Expected Impact:** +2-4% success rate
**Risk:** Low (filtering marginal outputs)
**Time:** 5 minutes

---

## Projected Success Rate

### Current (Phase 3)
**42.9% success rate**

### After Phase 4 (All improvements)

| Improvement | Impact | Cumulative |
|-------------|--------|------------|
| Baseline | - | 42.9% |
| + Opacity validation | +2-3% | 45-46% |
| + Blacklist V27/V142 | +3-5% | 48-51% |
| + Quality gate 200K | +2-4% | **50-55%** |

**🎯 PROJECTED: 50-55% success rate**

This would:
- ✅ Reach original Phase 3 target (55%)
- ✅ Achieve mid-tier competitive performance
- ✅ Improve from baseline by +7-12 percentage points

---

## Success Metrics for Phase 4

### After 4-6 Hours (30-50 submissions)

**✅ Success Criteria:**
- Success rate ≥50%
- No opacity corruption in logs
- No submissions to V27 or V142
- Average gaussian count ≥250K

**⚠️ Acceptable:**
- Success rate 45-49%
- Occasional opacity corruption caught by validation
- Some marginal improvements

**❌ Failure:**
- Success rate <45%
- Opacity corruption still causing rejections
- New issues emerge

---

## Long-Term Strategy

### Phase 5: Polish & Optimization (If Phase 4 successful)

**Additional improvements to consider:**
1. **Enable CLIP Validation** (--enable-validation=True)
   - Expected: +5-10% success rate
   - Risk: +0.5-1s generation time

2. **Prompt Enhancement** (--enable-prompt-enhancement=True)
   - Expected: +3-5% success rate
   - Risk: Minimal

3. **Fine-tune TRELLIS parameters**
   - Adjust CFG strength based on validator patterns
   - Test different step counts

4. **Image Enhancement** (--enable-image-enhancement=True)
   - Expected: +2-4% success rate
   - Risk: +0.5s generation time

**Target:** 65-75% success rate (top-tier competitive)

---

## Conclusion

### Key Takeaways

1. **Phase 3 achieved modest improvement** (+4.4% vs Phase 2)
2. **Discovered critical opacity bug** (14.3% of generations)
3. **CFG increase helped but wasn't magic bullet** (prompt adherence matters)
4. **Validator blacklisting is effective** (avoid V27, V142)
5. **Quality gate needs tuning** (200K better than 150K)

### The Path to 55% Success Rate

**Current:** 42.9%
**Phase 4 Target:** 50-55%
**Pathway:** Fix bugs + optimize validator selection + raise quality bar

**The Three Critical Fixes:**
1. ⚠️ Stop submitting invisible models (opacity validation)
2. ⚠️ Stop submitting to impossible validators (blacklist V27/V142)
3. ⚠️ Stop submitting low-density outputs (200K threshold)

**These are all defensive improvements** - preventing guaranteed failures rather than improving quality. Once implemented, we can focus on offensive improvements (CLIP validation, prompt enhancement) to push toward 65-75%.

---

**Analysis Complete**
**Next Action:** Implement Phase 4A (Opacity Validation)
**Timeline:** 10 minutes to implement, 4-6 hours to validate
**Expected Result:** 50-55% success rate
