# 🚀 404-GEN Competitive Miner - Production Configuration Guide

## Quick Start

```bash
# 1. Deploy competitive miner (one command)
./deploy_competitive.sh

# 2. Monitor logs
tail -f logs/generation_service.log
tail -f logs/miner.log

# 3. Run tests
./test_competitive.py

# 4. Stop everything
./stop_competitive.sh
```

---

## Configuration Options

### Generation Service (`serve_competitive.py`)

**Command Line Arguments:**

```bash
python generation/serve_competitive.py \
    --port 10006 \                      # Service port
    --config configs/text_mv_fast.yaml \ # DreamGaussian config
    --flux-steps 4 \                    # FLUX inference steps (1-8)
    --validation-threshold 0.6 \        # CLIP threshold (0.6 = minimum)
    --enable-validation                 # Enable CLIP pre-validation
```

**Recommended Settings by Hardware:**

| GPU | FLUX Steps | Config | Expected Speed |
|-----|-----------|--------|---------------|
| RTX 4090 | 4 | text_mv_fast.yaml | 15-20s |
| RTX 3090 | 4 | text_mv_fast.yaml | 20-25s |
| RTX 3080 | 2 | text_mv_ultra_fast.yaml | 15-20s |
| RTX 3070 | 2 | text_mv_ultra_fast.yaml | 20-30s |

**FLUX Steps Guide:**
- `1 step`: ~1s, lower quality (only for extremely fast hardware)
- `2 steps`: ~2s, acceptable quality
- `4 steps`: ~3s, **RECOMMENDED** (best quality/speed balance)
- `8 steps`: ~6s, marginal quality improvement (not worth it)

**Validation Threshold:**
- `0.6`: Network minimum (recommended to keep this)
- `0.7`: Higher quality, but may reject some valid results
- `0.8`: Very strict, only for testing

---

## Deployment Script Options

```bash
./deploy_competitive.sh [OPTIONS]

Options:
  --port PORT                      Service port (default: 10006)
  --flux-steps STEPS              FLUX inference steps (default: 4)
  --validation-threshold THRESH   CLIP threshold (default: 0.6)
  --config CONFIG                 DreamGaussian config (default: configs/text_mv_fast.yaml)
  --skip-deps                     Skip dependency installation
```

**Examples:**

```bash
# Standard deployment (recommended)
./deploy_competitive.sh

# Fast deployment on RTX 3080
./deploy_competitive.sh --flux-steps 2 --config configs/text_mv_ultra_fast.yaml

# Custom port
./deploy_competitive.sh --port 10007

# Skip dependency check (if already installed)
./deploy_competitive.sh --skip-deps
```

---

## DreamGaussian Configs

Three configs are available in `generation/configs/`:

### 1. `text_mv_fast.yaml` (RECOMMENDED)
```yaml
iters: 250              # 6x faster than base
ref_size: 128           # 4x faster VAE
guidance_scale: 50      # Faster convergence
batch_size: 4           # 4 MVDream views at once
num_pts: 4000          # 2x faster point processing
density_end_iter: 200  # 80% of total iters
```
**Speed:** ~15s for 3D generation
**Quality:** Excellent (optimized for CLIP >0.7)
**Use case:** Production (RTX 4090, 3090)

### 2. `text_mv_ultra_fast.yaml`
```yaml
iters: 150             # 10x faster than base
ref_size: 96           # 6x faster VAE
guidance_scale: 30     # Fastest convergence
batch_size: 4
num_pts: 3000         # 3x faster point processing
density_end_iter: 120 # 80% of total iters
```
**Speed:** ~10s for 3D generation
**Quality:** Good (CLIP ~0.65-0.7)
**Use case:** Lower-end GPUs (RTX 3080, 3070)

### 3. `text_mv.yaml` (Original optimized)
```yaml
iters: 1500            # High quality
ref_size: 256          # Best quality
guidance_scale: 120    # Maximum detail
batch_size: 1          # Conservative
num_pts: 8000         # Maximum points
density_end_iter: 1200
```
**Speed:** ~150s for 3D generation
**Quality:** Excellent (CLIP >0.8)
**Use case:** Testing/validation only (TOO SLOW for production)

---

## Expected Performance

### Pipeline Breakdown (RTX 4090, FLUX=4, fast config)

| Stage | Time | Component |
|-------|------|-----------|
| Text-to-Image | 3s | FLUX.1-schnell (4 steps) |
| Background Removal | 0.2s | BRIA RMBG 2.0 |
| 3D Generation | 15s | DreamGaussian (250 iters) |
| CLIP Validation | 0.5s | CLIP ViT-B/32 |
| **Total** | **~20s** | Full pipeline |

### Throughput Estimates

**4-hour window (240 minutes):**

| Setup | Time/Task | Tasks/4h | Notes |
|-------|----------|----------|-------|
| Base miner | 150s | ~9 | TOO SLOW |
| With speed opts | 25s | ~34 | Better but not competitive |
| **Competitive (fast)** | **20s** | **~120** | **RECOMMENDED** |
| Competitive (ultra-fast) | 15s | ~160 | Lower quality |

**Assuming 120s cooldown between validators, 50% task availability**

---

## Monitoring & Debugging

### Health Checks

```bash
# Check generation service health
curl http://localhost:10006/health | python -m json.tool

# Check stats
curl http://localhost:10006/stats | python -m json.tool

# Test generation
curl -X POST http://localhost:10006/generate/ \
     -F "prompt=a red cube" \
     -o test.ply
```

### Log Files

```bash
# Watch generation service logs
tail -f logs/generation_service.log

# Watch miner logs
tail -f logs/miner.log

# Search for errors
grep ERROR logs/*.log

# Search for validation failures
grep "Validation failed" logs/generation_service.log
```

### Key Metrics to Monitor

**Generation Service:**
- ✅ Generation time: Should be <25s
- ✅ CLIP scores: Should be >0.7
- ✅ File sizes: Should be 100-500KB
- ❌ Validation failures: Should be <10%

**Miner:**
- ✅ Tasks pulled: Should increase steadily
- ✅ Tasks submitted: Should match pulled
- ✅ Validator feedback: Check reward scores
- ❌ "No available validators": Check blacklist/stake settings

### Common Issues

**Issue:** "No GPU detected"
- Check: `nvidia-smi`
- Fix: Install CUDA drivers, restart

**Issue:** "Generation service failed to start"
- Check: `logs/generation_service.log`
- Common causes: Out of VRAM, missing dependencies
- Fix: Restart with `./deploy_competitive.sh`

**Issue:** "Validation failed" frequently
- Check: CLIP scores in logs
- Fix: Lower validation threshold to 0.6 or reduce FLUX steps to 4

**Issue:** "No available validators"
- Check: `logs/miner.log` for skip reasons
- Fix: Lower `min_stake_to_set_weights` in config or check blacklist

**Issue:** Generation too slow (>30s)
- Fix: Use ultra_fast config or reduce FLUX steps to 2

---

## Optimization Tuning

### Speed vs Quality Trade-offs

**Priority: Maximum Speed**
```bash
./deploy_competitive.sh \
    --flux-steps 2 \
    --config configs/text_mv_ultra_fast.yaml \
    --validation-threshold 0.6
```
- Speed: ~15s
- Quality: CLIP ~0.65-0.7
- Use case: High competition, need volume

**Priority: Maximum Quality**
```bash
./deploy_competitive.sh \
    --flux-steps 4 \
    --config configs/text_mv_fast.yaml \
    --validation-threshold 0.7
```
- Speed: ~20s
- Quality: CLIP ~0.7-0.8
- Use case: Lower competition, want better ELO

**Balanced (RECOMMENDED)**
```bash
./deploy_competitive.sh
```
- Speed: ~20s
- Quality: CLIP ~0.7+
- Use case: Production

### Advanced Tuning

**Increase concurrent workers** (if you have multiple generation endpoints):

Edit `neurons/miner/competitive_miner.py`:
```python
self.task_manager = AsyncTaskManager(
    max_concurrent_tasks=8,  # Increase from 4
    max_queue_size=40,       # Double queue size
    pull_interval=5.0        # Poll more frequently
)
```

**Adjust CLIP model** (in `generation/validators/clip_validator.py`):
```python
# Faster but less accurate
CLIPValidator(model_name="ViT-B/16")  # Default is ViT-B/32

# Slower but more accurate
CLIPValidator(model_name="ViT-L/14")
```

---

## Deployment Checklist

### Pre-Deployment

- [ ] GPU drivers installed (`nvidia-smi` works)
- [ ] Python 3.8+ installed
- [ ] At least 100GB free disk space (for models)
- [ ] At least 24GB VRAM (or 16GB with optimizations)
- [ ] Wallet registered on subnet 17
- [ ] `logs/` directory exists

### First Deployment

```bash
# 1. Clone/update repo
cd /path/to/three-gen-subnet

# 2. Create logs directory
mkdir -p logs

# 3. Deploy
./deploy_competitive.sh

# 4. Wait for models to load (2-3 minutes)
# Watch logs for "COMPETITIVE MINER READY FOR PRODUCTION"

# 5. Run tests
./test_competitive.py

# 6. Monitor for 15 minutes
tail -f logs/miner.log

# 7. Check feedback
grep "Feedback from" logs/miner.log
```

### Ongoing Monitoring

**Daily:**
- Check logs for errors
- Verify tasks being pulled and submitted
- Check CLIP scores and ELO

**Weekly:**
- Review validator feedback
- Adjust configuration if needed
- Update blacklist if new WC validators appear

---

## Upgrading

### From Base Miner

If you're currently running the base miner:

```bash
# 1. Stop base miner
pkill -f serve_miner.py
pkill -f serve.py

# 2. Deploy competitive
./deploy_competitive.sh

# 3. Verify
./test_competitive.py
```

### From Earlier Competitive Version

```bash
# 1. Stop current version
./stop_competitive.sh

# 2. Pull latest code
git pull

# 3. Redeploy
./deploy_competitive.sh --skip-deps  # If dependencies already installed
```

---

## Future Upgrades

### LGM (Large Gaussian Model)

User researched LGM as a potential upgrade:
- 3x faster than DreamGaussian
- Native Gaussian Splat output
- Feed-forward (consistent timing)

**When to upgrade:**
- If current system is still not competitive
- If LGM models become more available
- If speed <15s becomes critical

**Decision:** Deploy current system first, upgrade to LGM only if needed.

---

## Support

**Issues:**
- Check `COMPETITIVE_IMPLEMENTATION_COMPLETE.md` for architecture details
- Check logs in `logs/` directory
- Test with `./test_competitive.py`

**Performance Questions:**
- Expected: 120+ tasks per 4h
- Expected CLIP: >0.7
- Expected speed: 15-25s

**Questions about competitive requirements:**
- See Discord FAQ (referenced in implementation)
- UID 180 should be blacklisted
- CLIP validation should prevent cooldown penalties
