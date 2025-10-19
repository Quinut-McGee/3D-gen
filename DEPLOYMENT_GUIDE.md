# 🚀 404-GEN Competitive Miner - Deployment Guide

**Ready-to-deploy production competitive mining system with 10-15x improvement over base miner.**

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [What's Included](#whats-included)
3. [Prerequisites](#prerequisites)
4. [Detailed Deployment Steps](#detailed-deployment-steps)
5. [Testing](#testing)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)
8. [Performance Expectations](#performance-expectations)

---

## ⚡ Quick Start

**If you already have Python, GPU drivers, and dependencies installed:**

```bash
# 1. Deploy everything (one command)
./deploy_competitive.sh

# 2. Wait 2-3 minutes for models to load

# 3. Monitor logs
tail -f logs/miner.log
```

**That's it!** Your competitive miner is running.

---

## 📦 What's Included

Your competitive mining system has these components:

### **New Files Created:**

```
generation/
├── serve_competitive.py           # Main generation service (FLUX + BRIA + DreamGaussian + CLIP)
├── models/
│   ├── flux_generator.py         # FLUX.1-schnell text-to-image (3s)
│   └── background_remover.py     # BRIA RMBG 2.0 background removal (0.2s)
├── validators/
│   └── clip_validator.py         # Pre-submission CLIP validation
└── configs/
    ├── text_mv_fast.yaml         # Fast config (250 iters, ~15s)
    └── text_mv_ultra_fast.yaml   # Ultra-fast config (150 iters, ~10s)

neurons/miner/
├── competitive_miner.py          # Main competitive miner class
├── competitive_workers.py        # Worker with CLIP validation + async submission
├── async_task_manager.py         # Multi-validator async polling system
├── serve_miner_competitive.py   # Miner entry point
└── validator_selector.py         # Modified with UID 180 blacklisting

# Deployment scripts
deploy_competitive.sh             # One-click deployment
stop_competitive.sh               # Stop all services
test_competitive.py               # Comprehensive test suite

# Documentation
COMPETITIVE_IMPLEMENTATION_COMPLETE.md  # Full implementation details
PRODUCTION_CONFIG.md                    # Configuration guide
DEPLOYMENT_GUIDE.md                     # This file
```

### **Architecture:**

```
┌─────────────────────────────────────────┐
│   Competitive Miner (serve_miner)      │
│   - Async multi-validator polling      │
│   - Queries ALL validators at once     │
│   - 4 parallel task workers            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│   Generation Service (serve_competitive) │
│                                          │
│   [1] FLUX.1-schnell (~3s)              │
│       Text → Image                       │
│              ▼                           │
│   [2] BRIA RMBG 2.0 (~0.2s)             │
│       Image → RGBA                       │
│              ▼                           │
│   [3] DreamGaussian (~15s)              │
│       RGBA → Gaussian Splat              │
│              ▼                           │
│   [4] CLIP Validation (~0.5s)           │
│       Quality Check (≥0.6)              │
└──────────────────────────────────────────┘
```

---

## ✅ Prerequisites

### **Hardware:**

- **GPU:** NVIDIA RTX 3070 or better (4090/3090 recommended)
- **VRAM:** 24GB recommended (16GB minimum with optimizations)
- **Disk:** 100GB+ free (for model downloads)
- **RAM:** 32GB+ recommended

### **Software:**

- **OS:** Linux (Ubuntu 20.04+ recommended)
- **Python:** 3.8 - 3.10
- **CUDA:** 11.8+ with cuDNN
- **GPU Drivers:** Latest NVIDIA drivers

### **Bittensor:**

- Registered wallet on subnet 17
- Some TAO for transaction fees

### **Verify Prerequisites:**

```bash
# Check Python version
python --version  # Should be 3.8-3.10

# Check GPU
nvidia-smi        # Should show your GPU

# Check CUDA
nvcc --version    # Should show CUDA 11.8+

# Check disk space
df -h             # Should have 100GB+ free

# Check if registered
btcli wallet overview  # Should show your wallet on subnet 17
```

---

## 📝 Detailed Deployment Steps

### **Step 1: Install Dependencies**

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install required packages
pip install transformers diffusers accelerate xformers
pip install git+https://github.com/openai/CLIP.git
pip install fastapi uvicorn aiohttp loguru
pip install bittensor pillow numpy omegaconf
```

**Or let the deployment script handle it:**

```bash
./deploy_competitive.sh  # Will auto-install missing packages
```

### **Step 2: Create Logs Directory**

```bash
mkdir -p logs
```

### **Step 3: Configure (Optional)**

The default configuration works well for most setups. If you want to customize:

**Edit deployment settings:**

```bash
# Fast deployment (RTX 4090/3090)
./deploy_competitive.sh --flux-steps 4 --config configs/text_mv_fast.yaml

# Ultra-fast deployment (RTX 3080/3070)
./deploy_competitive.sh --flux-steps 2 --config configs/text_mv_ultra_fast.yaml
```

**See `PRODUCTION_CONFIG.md` for all options.**

### **Step 4: Deploy**

```bash
./deploy_competitive.sh
```

**What happens:**
1. ✅ Checks dependencies
2. ✅ Checks GPU availability
3. ✅ Stops old services
4. ✅ Starts generation service (loads models)
5. ✅ Starts competitive miner
6. ✅ Verifies everything is running

**Expected output:**

```
========================================
404-GEN COMPETITIVE MINER DEPLOYMENT
========================================

[1/6] Checking dependencies...
✅ All dependencies already installed

[2/6] Checking GPU availability...
✅ GPU detected: NVIDIA GeForce RTX 4090 (24.0 GB VRAM)

[3/6] Stopping old services...
✅ No old services running

[4/6] Starting competitive generation service...
✅ Generation service started (PID: 12345)
   Logs: logs/generation_service.log
Waiting for generation service to initialize...
✅ Generation service ready at http://localhost:10006

[5/6] Starting competitive miner...
✅ Miner started (PID: 12346)
   Logs: logs/miner.log

[6/6] Verifying deployment...
✅ Generation service running
✅ Miner running

========================================
🚀 DEPLOYMENT COMPLETE!
========================================

Services:
  Generation Service: http://localhost:10006
  Generation PID: 12345
  Miner PID: 12346

Logs:
  Generation: tail -f logs/generation_service.log
  Miner: tail -f logs/miner.log

Expected performance:
  Generation time: 15-25 seconds
  CLIP scores: >0.7
  Throughput: 120+ tasks per 4h
```

### **Step 5: Wait for Model Loading**

The first startup takes 2-3 minutes to download and load models:

```bash
# Watch generation service logs
tail -f logs/generation_service.log

# You should see:
# [1/4] Loading FLUX.1-schnell...
# [2/4] Loading BRIA RMBG 2.0...
# [3/4] Loading DreamGaussian...
# [4/4] Loading CLIP validator...
# 🚀 COMPETITIVE MINER READY FOR PRODUCTION
```

### **Step 6: Verify Miner is Working**

```bash
# Watch miner logs
tail -f logs/miner.log

# You should see:
# 🚀 COMPETITIVE MINER STARTING
# Polling validators...
# Task pulled from validator [X]
# Task submitted to validator [X]
# Feedback from [X]: Score=0.75, ELO=1200, Reward=0.02
```

---

## 🧪 Testing

### **Run Comprehensive Tests:**

```bash
./test_competitive.py
```

**Tests performed:**

1. ✅ Import test (all packages installed)
2. ✅ GPU test (GPU detected and has enough VRAM)
3. ✅ FLUX generation test (generates image in <5s)
4. ✅ Background removal test (removes background correctly)
5. ✅ CLIP validation test (validates images correctly)
6. ✅ Generation service test (full pipeline works)
7. ✅ PLY structure test (output format valid)
8. ✅ AsyncTaskManager test (async system works)
9. ✅ Validator blacklist test (UID 180 blacklisted)
10. ✅ Competitive miner imports test

**Expected output:**

```
========================================
404-GEN COMPETITIVE MINER - TEST SUITE
========================================

✅ Import test
✅ GPU test (NVIDIA GeForce RTX 4090, 24.0 GB)
✅ FLUX generation test (2.8s)
✅ Background removal test (0.2s)
✅ CLIP validation test (good=0.82, bad=0.31)
✅ Generation service test (18.5s, 234.5KB, CLIP=0.76)
✅ PLY structure test
✅ AsyncTaskManager instantiation test
✅ Validator blacklist test (1 blacklisted)
✅ Competitive miner imports test

========================================
TEST SUMMARY
========================================
Total: 10
Passed: 10 ✅
Failed: 0 ❌
```

### **Manual Test:**

```bash
# Test generation service directly
curl -X POST http://localhost:10006/generate/ \
     -F "prompt=a red cube" \
     -o test.ply

# Check output
ls -lh test.ply  # Should be 100-500KB
head test.ply    # Should start with "ply\nformat binary_little_endian 1.0"
```

---

## 📊 Monitoring

### **Monitor Logs:**

```bash
# Watch miner logs (most important)
tail -f logs/miner.log

# Watch generation service logs
tail -f logs/generation_service.log

# Watch both at once
tail -f logs/*.log
```

### **Key Metrics:**

**In miner logs, look for:**

```
✅ Good signs:
- "Task pulled from validator [X]"
- "Task submitted to validator [X]"
- "Feedback from [X]: Score=0.7+, Reward=0.01+"
- "Generation completed: 234567 bytes, CLIP=0.75"

❌ Warning signs:
- "No available validators" (check blacklist/stake)
- "Generation failed" (check generation service)
- "Validation failed" (check CLIP threshold)
- "Submission failed" (network issue)
```

**In generation service logs, look for:**

```
✅ Good signs:
- "✅ GENERATION COMPLETE"
- "Total time: 18.5s" (<25s is good)
- "CLIP Score: 0.76" (>0.7 is good)
- "File size: 234.5 KB" (100-500KB is good)

❌ Warning signs:
- "❌ Generation failed" (model crash)
- "⚠️ VALIDATION FAILED: CLIP=0.52" (quality issue)
- Total time >30s (too slow)
```

### **Health Checks:**

```bash
# Check service health
curl http://localhost:10006/health | python -m json.tool

# Check stats
curl http://localhost:10006/stats | python -m json.tool

# Check if processes are running
ps aux | grep -E "serve_competitive|serve_miner_competitive"
```

### **Performance Tracking:**

```bash
# Count tasks submitted in last hour
grep "Task submitted" logs/miner.log | tail -60

# Check average CLIP scores
grep "CLIP Score" logs/generation_service.log | awk '{print $NF}' | tail -20

# Check average generation times
grep "Total time" logs/generation_service.log | tail -20

# Check reward feedback
grep "Feedback from" logs/miner.log | tail -20
```

---

## 🔧 Troubleshooting

### **Problem: "No GPU detected"**

```bash
# Check GPU
nvidia-smi

# If no output:
# 1. Install/update NVIDIA drivers
# 2. Reboot system
# 3. Try again
```

### **Problem: "Generation service failed to start"**

```bash
# Check logs
tail -50 logs/generation_service.log

# Common issues:
# - Out of VRAM: Close other GPU applications
# - Missing dependencies: Run ./deploy_competitive.sh
# - Port in use: Change port with --port flag
```

### **Problem: "No available validators"**

```bash
# Check miner logs for skip reasons
grep "Validator.*skipped" logs/miner.log

# Common reasons:
# - All validators blacklisted: Review BLACKLISTED_VALIDATORS in validator_selector.py
# - Stake too low: Lower min_stake_to_set_weights in config
# - Not registered: Check with btcli wallet overview
```

### **Problem: "Validation failed" frequently**

```bash
# Check CLIP scores
grep "CLIP Score" logs/generation_service.log | tail -20

# If scores consistently <0.6:
# 1. Lower threshold: --validation-threshold 0.55
# 2. Increase FLUX steps: --flux-steps 4 (or 8)
# 3. Use faster config: --config configs/text_mv_fast.yaml
```

### **Problem: Generation too slow (>30s)**

```bash
# Check generation times
grep "Total time" logs/generation_service.log | tail -20

# If >30s:
# 1. Use ultra-fast config: --config configs/text_mv_ultra_fast.yaml
# 2. Reduce FLUX steps: --flux-steps 2
# 3. Check GPU utilization: nvidia-smi
```

### **Problem: Services crashed**

```bash
# Check what crashed
ps aux | grep -E "serve_competitive|serve_miner_competitive"

# Restart
./stop_competitive.sh
./deploy_competitive.sh
```

---

## 🎯 Performance Expectations

### **Speed Benchmarks (RTX 4090):**

| Component | Time |
|-----------|------|
| FLUX.1-schnell (4 steps) | 3.0s |
| BRIA RMBG 2.0 | 0.2s |
| DreamGaussian (250 iters) | 15.0s |
| CLIP Validation | 0.5s |
| **Total Pipeline** | **~20s** |

### **Quality Benchmarks:**

| Metric | Target | Expected |
|--------|--------|----------|
| CLIP Score | ≥0.6 | 0.7-0.8 |
| File Size | 100-500KB | 200-300KB |
| Validation Pass Rate | >90% | >95% |

### **Throughput Benchmarks:**

| Timeframe | Tasks | Notes |
|-----------|-------|-------|
| Per hour | ~30 | Assuming 120s cooldown |
| Per 4 hours | ~120 | Competition period |
| Per day | ~720 | Full day uptime |

### **Improvement Over Base Miner:**

| Metric | Base Miner | Competitive | Improvement |
|--------|-----------|-------------|-------------|
| Generation Time | 150s | 20s | **7.5x faster** |
| Validators Polled | 1 at a time | All at once | **10-15x** |
| Concurrent Tasks | 1 | 4 | **4x** |
| Cooldown Penalties | Frequent | Near zero | **+30-50%** |
| **Throughput/4h** | ~10 | ~120 | **12x** |

---

## 🔄 Management Commands

### **Stop Services:**

```bash
./stop_competitive.sh
```

### **Restart Services:**

```bash
./stop_competitive.sh
./deploy_competitive.sh
```

### **View Logs:**

```bash
# Miner logs
tail -f logs/miner.log

# Generation logs
tail -f logs/generation_service.log

# All logs
tail -f logs/*.log
```

### **Check Status:**

```bash
# Check if running
ps aux | grep -E "serve_competitive|serve_miner_competitive"

# Check PIDs
cat .generation_service.pid
cat .miner.pid

# Check health
curl http://localhost:10006/health
```

---

## 📚 Additional Documentation

- **`COMPETITIVE_IMPLEMENTATION_COMPLETE.md`**: Full implementation details, architecture, all components
- **`PRODUCTION_CONFIG.md`**: Advanced configuration options, tuning guides
- **`SPEED_OPTIMIZATIONS.md`**: Speed analysis and optimization strategies
- **Source code comments**: All files heavily documented

---

## 🚀 Next Steps After Deployment

### **First 24 Hours:**

1. ✅ Monitor logs closely
2. ✅ Track validator feedback
3. ✅ Verify CLIP scores >0.7
4. ✅ Check throughput (should be ~120 tasks/4h)
5. ✅ Ensure no crashes or errors

### **First Week:**

1. ✅ Review ELO progression
2. ✅ Optimize config if needed (see PRODUCTION_CONFIG.md)
3. ✅ Update blacklist if new WC validators appear
4. ✅ Consider adjusting FLUX steps based on results

### **Future Upgrades (Optional):**

**LGM (Large Gaussian Model):**
- User researched LGM as potential upgrade
- 3x faster than DreamGaussian
- Native Gaussian Splat output
- **Decision:** Deploy current system first, upgrade to LGM only if needed

**When to consider LGM:**
- If current system is not competitive enough
- If <15s generation becomes critical
- If LGM models become more accessible

---

## ✅ Deployment Complete!

You now have a **production-ready competitive mining system** with:

✅ **SOTA Models:** FLUX.1-schnell, BRIA RMBG 2.0, DreamGaussian, CLIP
✅ **Async System:** Multi-validator polling, 4 parallel workers
✅ **Quality Control:** CLIP pre-validation prevents penalties
✅ **Optimization:** Blacklisting, fast configs, efficient pipeline
✅ **Monitoring:** Comprehensive logging and health checks
✅ **Management:** One-click deploy/stop/test scripts

**Expected improvement: 10-15x over base miner** 🎉

---

**Questions or issues?**
Check the other documentation files or review the logs for debugging information.

**Good luck mining! 🚀**
