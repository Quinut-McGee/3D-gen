404-GEN Bittensor Miner - Project Documentation

  ## Project Overview

  This is a competitive Bittensor Subnet 17 (404-GEN) miner
  that generates high-quality 3D Gaussian Splat models from
  text prompts. The subnet rewards miners who can produce
  accurate, detailed 3D models that validators accept.

  **Current Status:**
  - **Network:** Bittensor Mainnet (Subnet 17)
  - **UID:** 57
  - **Rank:** 220 out of ~240 active miners
  - **Trust:** 1.0000
  - **Emission:** 0.352092
  - **Target:** Move to top 150 (mid-tier competitive)

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

  Phase 4: TRELLIS Opacity Fix (CURRENT)

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
  - Status: Deployed and stable, ranked 220/240

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

  Option A: Prompt Enhancement (Priority 1)

  - Goal: Enhance sparse prompts with descriptive details
  - Expected impact: +15-20% success rate (55% → 70-75%)
  - Cost: None (software only)
  - Risk: Very low
  - Implementation: 30 minutes

  Option B: Image Enhancement (Priority 2)

  - Goal: Re-enable --enable-image-enhancement=True
  - Details: Sharpness 3.5x, Contrast 1.8x, Detail filter
  - Expected impact: +10-15% success rate
  - Cost: +0.5-1s generation time
  - Risk: Low (proven settings)
  - Implementation: 1 command

  Option C: Quality Gate with Retry (Priority 3)

  - Goal: Enable --min-gaussian-count=200000 or 400000
  - Details: Filter sparse generations, retry with enhanced
  parameters
  - Expected impact: +20-25% success rate (by filtering
  failures)
  - Trade-off: May reduce throughput
  - Risk: Low (existing code)
  - Implementation: 1 command

  Option D: Increase TRELLIS Sampling Steps (If needed)

  - Goal: Increase sampling steps in serve_trellis.py (60→80,
   50→60)
  - Expected impact: +10-20% success rate, +50K-100K
  gaussians
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

  Opacity Fix Applied

  - File: /home/kobe/404-gen/v1/3D-gen/generation/DreamGaussi
  anLib/GaussianSplattingModel.py
  - Line: 269
  - Change: Widened clamp range from [0.001, 0.999] to
  [0.0001, 0.9999]
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
  - Prioritize low-risk, high-reward optimizations (prompt
  enhancement > image enhancement > quality gates > sampling
  steps)
  - If in doubt, ask the user before deploying to production

  Current focus: Diagnose overnight performance to validate
  consistency hypothesis, then deploy targeted improvements
  to reach 75-80% success rate.
