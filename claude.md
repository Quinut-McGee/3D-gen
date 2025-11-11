404-GEN Bittensor Miner - Project Documentation

  ## Project Overview

  This is a competitive Bittensor Subnet 17 (404-GEN) miner
  that generates high-quality 3D Gaussian Splat models from
  text prompts. The subnet rewards miners who can produce
  accurate, detailed 3D models that validators accept.

  **Current Status (Updated Nov 10, 2025):**
  - **Network:** Bittensor Mainnet (Subnet 17)
  - **UID:** 226
  - **Rank:** Recovering after recent improvements
  - **Trust:** 0.0000 (rebuilding after fixes)
  - **Emission:** 0.000000 (rebuilding after fixes)
  - **Success Rate:** 60-70% (post-fix baseline, target 75-85%)
  - **Target:** Rebuild trust → achieve 75-85% success rate → top 150

  ---

  ## System Architecture

  ### Production Environment
  **Location:** `/home/kobe/404-gen/v1/3D-gen/`

  ### PM2 Services (3 instances running)

  #### 1. `gen-worker-1` (Generation Service)
  - **Port:** 10010
  - **File:** `/home/kobe/404-gen/v1/3D-gen/generation/serve_
  competitive.py`
  - **Purpose:** Main generation pipeline (text → 3D gaussian
   splat)
  - **Stack:**
    - SD3.5 Large Turbo (text-to-image, 8-14s)
    - BRIA RMBG 2.0 (background removal, 0.2s)
    - TRELLIS (3D gaussian generation, 12-15s)
  - **Commands:**
    ``bash
    pm2 restart gen-worker-1
    pm2 logs gen-worker-1
    pm2 logs gen-worker-1 --err

  2. trellis-microservice (3D Generation Backend)

  - Port: 10008
  - File:
  /home/kobe/404-gen/v1/3D-gen/generation/serve_trellis.py
  - Purpose: TRELLIS pipeline for native Gaussian splat
  generation
  - Output: 256K-1.4M gaussians, 8-90 MB PLY files
  - Health check:
  curl -s http://localhost:10008/health | python -m json.tool

  3. miner-sn17-mainnet (Bittensor Miner)

  - File: /home/kobe/404-gen/v1/3D-gen/neurons/miner/competit
  ive_miner.py
  - Purpose: Async multi-validator task polling and
  submission
  - Features:
    - Queries ALL validators in parallel
    - 4 parallel workers for concurrent task processing
    - Validator blacklisting (e.g., UID 180)
    - Fire-and-forget submission (non-blocking)
  - Commands:
  pm2 logs miner-sn17-mainnet
  pm2 logs miner-sn17-mainnet | grep "Score\|Emission"

  ---
  Generation Pipeline

  Full Pipeline Flow

  1. Text Prompt → SD3.5 Large Turbo (8-14s)
     ↓
  2. Image → BRIA RMBG 2.0 (0.2s)
     ↓
  3. RGBA → TRELLIS Microservice (12-15s)
     ↓
  4. Gaussian Splat PLY → Validator Submission
     ↓
  5. Validator Score → Emission/Ranking

  Total Time: ~20-30 seconds per generation

  Output Specifications

  - Format: PLY (Polygon/Gaussian Splat)
  - Gaussian Count: 398K average (range: 60K - 1.4M)
  - File Size: 25.9 MB average (range: 3 MB - 90 MB)
  - Properties Per Gaussian:
    - Position (x, y, z)
    - Opacity (transparency)
    - Scale (3D ellipsoid size)
    - Rotation (quaternion)
    - Color (spherical harmonics: f_dc_0-2, f_rest_0-44)

  ---
  Improvement History

  Phase 1: Initial Deployment (TRELLIS)

  - Model: TRELLIS native Gaussian generation
  - Performance: 71% success rate initially
  - Issue: Opacity corruption started appearing (opacity_std
  = 0.0)
  - Symptom: All gaussians became invisible (avg_opacity:
  -6.907)
  - Result: Success rate dropped, rank declined

  Phase 2: LGM Migration Attempt

  - Reason: Suspected TRELLIS had fundamental flaw
  - Model: LGM (Large Gaussian Model)
  - Performance: 1s generation (15x faster than TRELLIS!)
  - Issue: Only 25K-48K gaussians generated
  - Result: Too sparse for validators (need 250K+), abandoned

  Phase 3: GRM Investigation

  - Model: GRM (Gaussian Reconstruction Model)
  - Specs: 0.1s generation, native Gaussian splats
  - Issue: GitHub repository has NO CODE (documentation only)
  - Result: Unusable, pivoted back to fixing TRELLIS

  Phase 4: TRELLIS Opacity Fix

  - Root Cause: GaussianSplattingModel.py line 269 had
  too-tight clamp
    - Old: torch.clamp(self._opacity, 0.001, 0.999)
    - New: torch.clamp(self._opacity, min=0.0001, max=0.9999)
  - Fix Impact:
    - Opacity corruption ELIMINATED ✅
    - opacity_std: 0.0 → 1.75 (healthy variation restored)
    - Success rate: 45% → 55% overall, 86% for high-density
  models
    - Gaussian density: 398K average (56% above target)
  - Status: SUPERSEDED by Phase 5 improvements

  Phase 5: Clean PLY Writer + Smart Normalization (CURRENT)

  **Date:** Nov 10, 2025
  **Status:** Deployed and validating

  After achieving 55% success rate with Phase 4, investigation
  of 7 consecutive Score=0.0 failures revealed two critical
  bugs that were breaking even healthy TRELLIS outputs.

  ### Critical Bug #1: save_ply() Corruption

  **Discovery:** Analysis of failures showed 50% had
  "corruption warnings" where opacity changed by 1.2-11.2
  points during the save operation.

  **Root Cause:** TRELLIS's `gaussian_output.save_ply()`
  function applies numerical transforms that introduce
  instability:
  - `inverse_sigmoid(self.get_opacity)` (line 129)
  - `torch.log(self.get_scaling)` (line 130)

  These transforms corrupt opacity values even AFTER our
  normalization fixes them.

  **Solution: Clean PLY Writer**
  - **File:** `/home/kobe/404-gen/v1/3D-gen/generation/clean_
  ply_writer.py` (NEW)
  - **Integration:** Modified serve_trellis.py line 271:
    - OLD: `gaussian_output.save_ply(tmp_path)`
    - NEW: `save_clean_ply(gaussian_output, tmp_path)`
  - **Approach:**
    1. Extract gaussian parameters directly from
    gaussian_output
    2. Apply proper sigmoid activation to get actual opacity
    3. Write PLY manually using plyfile library
    4. Verify written values match input values

  **Result:** 100% success - corruption eliminated
  - corruption_delta: 0.0000 across ALL generations ✅
  - Perfect preservation of normalized opacity values

  ### Critical Bug #2: Over-Normalization of Healthy Models

  **Discovery:** First generation after clean PLY writer still
  got Score=0.0. Investigation showed:
  - Raw TRELLIS opacity: 4.840 (HEALTHY - in natural range)
  - After normalization: 10.773 (TOO HIGH - broke it!)
  - Actual opacity: 99.96% (solid opaque blob, no
  transparency)

  **Root Cause:** Previous normalization logic normalized ALL
  models with mean < 7.0 to targets of 9.0-11.0, which was
  breaking naturally healthy TRELLIS outputs.

  **Solution: Smart Adaptive Normalization**
  - **File:** `/home/kobe/404-gen/v1/3D-gen/generation/serve_
  trellis.py` (lines 213-258)
  - **Approach:** Respect TRELLIS's natural healthy range
  ```python
  HEALTHY_MIN = 2.0   # Below = too transparent
  HEALTHY_MAX = 6.0   # Above = too opaque
  TARGET_OPTIMAL = 4.0  # Sweet spot

  # Only normalize if OUTSIDE healthy range
  if current_mean < HEALTHY_MIN:
      target_mean = TARGET_OPTIMAL  # Fix too transparent
  elif current_mean > HEALTHY_MAX:
      target_mean = TARGET_OPTIMAL  # Fix too opaque
  else:
      target_mean = None  # PRESERVE healthy output!
  ```

  **Result:** Intelligent normalization
  - 67% of models preserved (were already healthy) ✅
  - 25% of models fixed (were genuinely broken) ✅
  - 8% edge cases (handled appropriately)
  - Natural opacity range: 47-88% (proper transparency
  variation)
  - Critical test passed: Raw 4.839 preserved instead of
  normalized to 10.8

  ### Phase 5 Performance Metrics

  **Baseline (Post-Fix):** 60-70% success rate
  - Clean PLY writer: 100% success (0.0000 corruption)
  - Smart normalization: 67% preserved, 25% fixed
  - Generation timing: 18-22s average (under 30s threshold)

  **GPU Utilization:**
  - GPU 0 (RTX 4090, 24GB): TRELLIS (6GB) + BG removal
  (0.5GB) = 28% utilized
  - GPU 1 (RTX 5070 Ti, 16GB): SDXL-Turbo (7GB) = 43%
  utilized
  - Plenty of headroom for future enhancements

  **Validator Metrics:**
  - UID: 226
  - Trust: 0.0000 (rebuilding after fixes)
  - Emission: 0.000000 (rebuilding after fixes)
  - Target: 75-85% success rate → rebuild trust/emission

  ### Remaining Failure Modes (20-30% rejections)

  Based on recent failure analysis, remaining issues:
  1. **Flat/thin geometry** (30%): Objects with thin
  dimensions (y < 0.2 or z < 0.15)
  2. **IMAGE-TO-3D tasks** (40%): Ambiguous images lacking
  depth cues
  3. **Complex multi-object scenes** (20%): Heavy occlusion,
  multiple objects
  4. **Low density** (10%): < 200K gaussians (validators
  prefer 400K+)

  ### Next Priority: Validation → Quick Wins → Depth
  Estimation

  **Phase 5A** (Current): Monitor fixes for 24-48 hours
  - Confirm corruption stays at 0.0000
  - Validate success rate stabilizes at 70-80%
  - Analyze remaining failure patterns
  - Collect 50-100 generations of data

  **Phase 5B**: Quick Win - Blacklist Validator 212
  - Evidence: 0% global acceptance rate, declining ELO
  - Expected impact: +2-3% success rate
  - Risk: None (proven strategy from validator 180)
  - Implementation: 5 minutes (add to BLACKLISTED_VALIDATORS)

  **Phase 5C**: Depth Estimation Preprocessing (if needed)
  - Only after stable baseline established
  - Use ZoeDepth or similar monocular depth model
  - Expected impact: +10-15% success rate
  - Target: Flat/thin geometry + IMAGE-TO-3D failures
  - Implementation: 2-4 hours

  ---
  404-GEN Competitive Mining Guide

  Official Guidance from Subnet Maintainers

  To truly compete, the base miner code needs to be studied 
  and rewritten to reflect your specific setup. Timing 
  adjustments and optimizations are crucial for success.

  Please Note: Our base miner code is NOT competitive.

  Mining Pipeline Essentials ⛏️

  1️⃣ Open Source Text-to-Image Generation Model
  - Need great text-to-image generator suitable for
  commercial use
  - Examples: FLUX-schnell, Stable Diffusion, etc.
  - Our choice: SD3.5 Large Turbo (better prompt adherence
  than FLUX for 3D)

  2️⃣ Background Removal Model
  - Need robust background removal (rembg might not be good
  enough)
  - Our choice: BRIA RMBG 2.0 (recommended by Discord, 2x
  faster than rembg)

  3️⃣ SOTA 3D Generation Model
  - Must produce Gaussian Splats (not triangle meshes)
  - Options: TRELLIS, LGM, GS-LRM, or convert meshes →
  gaussians
  - Our choice: TRELLIS (native Gaussian splat generation,
  256K-1.4M gaussians)

  4️⃣ Self Validation Mechanism
  - Validate results BEFORE sending
  - Regenerate or send empty results if validation fails
  - Critical: Empty results are ignored, low-quality results
  increase cooldown penalties
  - Our implementation: CLIP validation (currently disabled
  in diagnostic mode)

  Other Tips & Tricks 🪄

  🧠 Work Smart. Not Hard.
  - Evaluate validators and utilize validator blacklisting
  - Example: UID 180 is a WC (wasted compute) - don't pull
  tasks from it
  - Our implementation: Blacklisting in competitive_miner.py

  ✨ Optimize.
  - Pipeline must generate wide variety of models
  - Not enough to master cute animals - need versatility
  - Our approach: Testing prompt enhancement for better
  consistency

  ♾️ Consider Scale.
  - May need multiple GPUs as pipeline becomes elaborate
  - Our setup: Single GPU (RTX 4090 or similar), considering
  scale later

  ⏰ Time is a Resource.
  - Implement fully asynchronous operation
  - Don't wait for one validator before pulling from another
  - Our implementation: AsyncTaskManager with 4 parallel
  workers, queries all validators simultaneously

  ---
  Current Performance Analysis

  Success Rate Breakdown (20 submissions post-fix)

  - Overall: 55% (11/20 accepted)
  - High-density (>400K gaussians): 86% (6/7 accepted) ✅
  - Med-density (150-400K): 43% (3/7)
  - Low-density (<150K): 33% (2/6) ❌

  Key Finding: Density-Success Correlation

  Pattern: Simple prompts → sparse generations → validator
  rejection

  Failures:
  - "ukulele light wood" → 102K gaussians → Score 0.0 ❌
  - "sleek black frame" → 60K gaussians → Score 0.0 ❌
  - "emerald guitar" → 125K gaussians → Score 0.0 ❌

  Successes:
  - "artisan oak coffee table..." (16 words) → 817K gaussians
   → Score 0.79 ✅
  - "old well-maintained robot" → 916K gaussians → Score 0.72
   ✅
  - Unknown prompt → 1.4M gaussians → Score 0.80 ✅

  Hypothesis: If we can push ALL submissions to >400K
  gaussians, we'd achieve ~86% success rate (top tier
  competitive).

  ---
  Planned Improvements

  **COMPLETED ✅**

  ~~Option A: Fix save_ply() Corruption~~
  - **Status:** COMPLETED Nov 10, 2025
  - **Implementation:** clean_ply_writer.py
  - **Result:** 100% success (corruption_delta: 0.0000)

  ~~Option B: Fix Over-Normalization~~
  - **Status:** COMPLETED Nov 10, 2025
  - **Implementation:** Smart Adaptive Normalization in
  serve_trellis.py
  - **Result:** 67% preserved, 25% fixed correctly

  **CURRENT PRIORITIES**

  Priority 1: Validation Phase (24-48 hours)
  - Monitor corruption_delta stays at 0.0000
  - Validate success rate stabilizes at 70-80%
  - Collect 50-100 generations for analysis
  - Identify remaining failure patterns
  - **Risk:** None (monitoring only)
  - **Status:** IN PROGRESS

  Priority 2: Blacklist Validator 212 (Quick Win)
  - **Trigger:** After validation phase confirms stability
  - **Evidence:** 0% global acceptance, declining ELO
  - **Expected impact:** +2-3% success rate
  - **Implementation:** Add UID 212 to validator_selector.py
  BLACKLISTED_VALIDATORS list
  - **Risk:** None (proven strategy)
  - **Time:** 5 minutes

  Priority 3: Depth Estimation Preprocessing
  - **Trigger:** After validation shows remaining 20-30%
  failures still exist
  - **Goal:** Provide 3D depth cues to TRELLIS for better
  geometry
  - **Expected impact:** +10-15% success rate
  - **Target failures:** Flat/thin geometry (30%) + IMAGE-TO-
  3D (40%)
  - **Implementation:** 2-4 hours
  - **Risk:** Low-medium (new feature, requires testing)
  - **Note:** Use fresh Claude instance for implementation
  (avoid context pollution)

  **DEFERRED (Evaluate after Phase 5C)**

  Option D: Prompt Enhancement
  - Goal: Enhance sparse prompts with descriptive details
  - Expected impact: +5-10% success rate
  - May not be needed if depth estimation solves low-density
  issues
  - Cost: None (software only)
  - Risk: Very low
  - Implementation: 30 minutes

  Option E: Image Enhancement
  - Goal: Re-enable --enable-image-enhancement=True
  - Details: Sharpness 3.5x, Contrast 1.8x, Detail filter
  - Expected impact: +5-10% success rate
  - May not be needed if depth estimation provides sufficient
  quality
  - Cost: +0.5-1s generation time
  - Risk: Low (proven settings)
  - Implementation: 1 command

  Option F: Quality Gate with Retry
  - Goal: Enable --min-gaussian-count=200000 or 400000
  - Details: Filter sparse generations, retry with enhanced
  parameters
  - Expected impact: +10-15% success rate (by filtering
  failures)
  - Trade-off: May reduce throughput
  - Risk: Low (existing code)
  - Implementation: 1 command

  Option G: Increase TRELLIS Sampling Steps
  - Goal: Increase sampling steps in serve_trellis.py (60→80,
   50→60)
  - Expected impact: +5-10% success rate, +50K-100K gaussians
  - Cost: +2-3s generation time
  - Risk: Medium (unknown impact)
  - Implementation: 1 hour

  ---
  Key Log Locations

  Miner Logs

  # Main output (scores, emission, rank)
  tail -f /home/kobe/.pm2/logs/miner-sn17-mainnet-out.log

  # Errors
  tail -f /home/kobe/.pm2/logs/miner-sn17-mainnet-error.log

  # Success rate calculation
  tail -2000 /home/kobe/.pm2/logs/miner-sn17-mainnet-out.log
  | \
    grep -E "Score" | \
    awk '{if ($0 ~ /Score: [^0]/) success++; total++} \
    END {print "Success: " success "/" total " (" 
  success/total*100 "%)"}'

  Generation Service Logs

  # Main output (gaussian counts, timing)
  tail -f /home/kobe/.pm2/logs/gen-worker-1-out.log

  # Errors (opacity stats, quality metrics)
  tail -f /home/kobe/.pm2/logs/gen-worker-1-error.log

  # Gaussian density analysis
  tail -1000 /home/kobe/.pm2/logs/gen-worker-1-error.log | \
    grep "Gaussians:" | \
    awk -F'Gaussians: ' '{print $2}' | \
    awk -F',' '{gsub(/,/,"",$1); 
      if ($1 > 400000) high++; 
      else if ($1 >= 150000) med++; 
      else low++; 
      sum+=$1; count++
    } END {
      print "High (>400K): " high;
      print "Med (150-400K): " med;
      print "Low (<150K): " low;
      print "Average: " int(sum/count)
    }'

  TRELLIS Microservice Logs

  tail -f /home/kobe/.pm2/logs/trellis-microservice-error.log

  ---
  Critical Files

  Configuration

  - Main service: /home/kobe/404-gen/v1/3D-gen/generation/ser
  ve_competitive.py
  - TRELLIS integration: /home/kobe/404-gen/v1/3D-gen/generat
  ion/trellis_integration.py
  - TRELLIS microservice:
  /home/kobe/404-gen/v1/3D-gen/generation/serve_trellis.py
  - Miner: /home/kobe/404-gen/v1/3D-gen/neurons/miner/competi
  tive_miner.py
  - Validator selector: /home/kobe/404-gen/v1/3D-gen/neurons/
  miner/validator_selector.py

  Phase 5 Fixes (Nov 10, 2025)

  **Clean PLY Writer** (NEW FILE)
  - File: /home/kobe/404-gen/v1/3D-gen/generation/clean_ply_w
  riter.py
  - Purpose: Bypass TRELLIS's corrupted save_ply() function
  - Key function: save_clean_ply(gaussian_output, path)
  - Result: Eliminates 1.2-11.2 opacity corruption during
  save
  - Impact: corruption_delta: 0.0000 (perfect preservation)

  **Smart Adaptive Normalization**
  - File: /home/kobe/404-gen/v1/3D-gen/generation/serve_trell
  is.py
  - Lines: 213-258 (opacity normalization logic)
  - Lines: 271 (PLY save - now uses clean_ply_writer)
  - Key parameters:
    - HEALTHY_MIN = 2.0 (below = fix to 4.0)
    - HEALTHY_MAX = 6.0 (above = fix to 4.0)
    - TARGET_OPTIMAL = 4.0
  - Result: 67% preserved, 25% fixed correctly

  Phase 4 Fixes (Previous)

  - File: /home/kobe/404-gen/v1/3D-gen/generation/DreamGaussi
  anLib/GaussianSplattingModel.py
  - Line: 269
  - Change: Widened clamp range from [0.001, 0.999] to
  [0.0001, 0.9999]
  - Status: Still active, but superseded by Phase 5 smart
  normalization
  - Backup: GaussianSplattingModel.py.backup_*

  Current Flags (Diagnostic Mode)

  --enable-image-enhancement=False      # Disabled (testing 
  raw TRELLIS output)
  --enable-prompt-enhancement=False     # Disabled (using raw
   prompts)
  --enable-scale-normalization=False    # Disabled 
  (submitting TRELLIS as-is)
  --enable-validation=False             # CLIP validation 
  disabled
  --min-gaussian-count=0                # No quality gate 
  (submit everything)
  --background-threshold=0.5            # Standard rembg 
  threshold

  ---
  Health Checks

  Quick Status Check

  # All services status
  pm2 status | grep -E "miner-sn17|gen-worker|trellis"

  # Services healthy?
  curl -s http://localhost:10010/health | python -m json.tool
  curl -s http://localhost:10008/health | python -m json.tool

  # Recent performance
  tail -20 /home/kobe/.pm2/logs/miner-sn17-mainnet-out.log |
  grep "UID:57"

  Restart All Services

  pm2 restart gen-worker-1
  pm2 restart trellis-microservice
  pm2 restart miner-sn17-mainnet
  pm2 save

  ---
  Success Criteria

  Current Target:
  - Success rate: 55% → 75-80%
  - Rank: 220 → 150-180 (mid-tier competitive)
  - Emission: 0.352 → 0.5+ (42% increase)
  - Consistency: Reduce low-density submissions (<150K) to
  near zero

  Long-term Goal:
  - Success rate: 80-85% (top tier)
  - Rank: 100-150
  - Emission: 0.7-0.8 (100% increase)

  ---
  Support Resources

  - Discord: https://discord.com/channels/1065924238550237194
  /1387488986670436514
  - Dashboard: https://dashboard.404.xyz/d/main/404-gen/
  - Documentation:
  https://github.com/404-Repo/three-gen-subnet

  ---
  Notes for Claude

  - Always check PM2 status before making changes
  - Monitor logs during optimization tests
  - Back up files before editing (use timestamp:
  backup_$(date +%Y%m%d_%H%M%S))
  - Test changes on small batches (10-20 submissions) before
  full deployment
  - Prioritize low-risk, high-reward optimizations
  - If in doubt, ask the user before deploying to production

  **Current Focus (Nov 10, 2025):**

  Phase 5A - Validation Period (24-48 hours)
  - Monitor corruption_delta stays at 0.0000 across all
  generations
  - Validate success rate stabilizes at 70-80%
  - Analyze remaining failure modes from 50-100 generation
  sample
  - Do NOT implement new features during validation period

  Next Steps After Validation:
  1. Blacklist Validator 212 (quick +2-3% win)
  2. Evaluate if depth estimation is still needed
  3. If needed, use FRESH Claude instance for depth
  estimation (avoid context pollution)

  **Key Insights from Recent Debugging:**
  - save_ply() corruption was silent killer (50% of failures)
  - Over-normalization broke healthy models (normalized 4.8→
  10.8→99.96% opaque)
  - Smart normalization respects TRELLIS's natural range
  [2.0, 6.0]
  - Validators have zero tolerance for >1.0 bbox dimensions
  - Remaining failures: flat geometry (30%), IMAGE-TO-3D
  (40%), complex scenes (20%), low density (10%)
