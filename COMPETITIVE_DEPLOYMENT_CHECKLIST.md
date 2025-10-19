# 🚀 Competitive Miner - Deployment Checklist

Use this checklist to ensure your deployment is successful.

---

## 📋 Pre-Deployment Checklist

### **Hardware Requirements**

- [ ] GPU: NVIDIA RTX 3070 or better
- [ ] VRAM: 24GB (or 16GB minimum)
- [ ] Disk space: 100GB+ free
- [ ] RAM: 32GB+

**Verify:**
```bash
nvidia-smi              # Check GPU and VRAM
df -h                   # Check disk space
free -h                 # Check RAM
```

### **Software Requirements**

- [ ] OS: Linux (Ubuntu 20.04+)
- [ ] Python: 3.8-3.10
- [ ] CUDA: 11.8+ with cuDNN
- [ ] GPU drivers: Latest NVIDIA drivers

**Verify:**
```bash
python --version        # Should be 3.8-3.10
nvcc --version          # Should show CUDA 11.8+
nvidia-smi              # Should show driver version
```

### **Bittensor Setup**

- [ ] Wallet created
- [ ] Wallet registered on subnet 17
- [ ] TAO for transaction fees

**Verify:**
```bash
btcli wallet overview   # Should show your wallet
```

### **Project Setup**

- [ ] Repository cloned/updated
- [ ] In correct directory (`three-gen-subnet`)
- [ ] Logs directory exists

**Verify:**
```bash
pwd                     # Should end with three-gen-subnet
ls -la logs/            # Should exist
mkdir -p logs           # Create if needed
```

---

## 🚀 Deployment Steps

### **Step 1: Install Dependencies**

- [ ] Virtual environment created (optional but recommended)
- [ ] PyTorch with CUDA installed
- [ ] All required packages installed

**Quick check:**
```bash
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

**Or let the script handle it:**
```bash
./deploy_competitive.sh  # Will auto-install missing packages
```

### **Step 2: Deploy Competitive Miner**

- [ ] Run deployment script
- [ ] Wait for model loading (2-3 minutes)
- [ ] Verify both services started

**Deploy:**
```bash
./deploy_competitive.sh
```

**Expected output:**
```
========================================
🚀 DEPLOYMENT COMPLETE!
========================================

Services:
  Generation Service: http://localhost:10006
  Generation PID: XXXXX
  Miner PID: YYYYY
```

### **Step 3: Verify Services**

- [ ] Generation service health check passes
- [ ] Miner is polling validators
- [ ] No errors in logs

**Verify:**
```bash
# Health check
curl http://localhost:10006/health | python -m json.tool

# Check processes
ps aux | grep -E "serve_competitive|serve_miner_competitive"

# Check logs for errors
tail -50 logs/generation_service.log | grep -i error
tail -50 logs/miner.log | grep -i error
```

---

## 🧪 Testing Checklist

### **Step 4: Run Tests**

- [ ] All 10 tests pass
- [ ] No critical warnings

**Run tests:**
```bash
./test_competitive.py
```

**Expected:**
```
========================================
TEST SUMMARY
========================================
Total: 10
Passed: 10 ✅
Failed: 0 ❌
```

### **Step 5: Manual Generation Test**

- [ ] Test generation completes successfully
- [ ] Generation time <25 seconds
- [ ] PLY file created (100-500KB)
- [ ] CLIP score >0.7

**Test:**
```bash
# Generate test model
time curl -X POST http://localhost:10006/generate/ \
     -F "prompt=a red cube" \
     -o test.ply

# Check output
ls -lh test.ply         # Should be 100-500KB
head test.ply           # Should start with "ply"
```

---

## 📊 Monitoring Checklist

### **Step 6: Monitor Initial Operation (15 minutes)**

- [ ] Miner is pulling tasks from validators
- [ ] Tasks are being generated successfully
- [ ] Tasks are being submitted to validators
- [ ] CLIP scores are >0.7
- [ ] Generation times are <25s
- [ ] No validation failures
- [ ] Validator feedback is being received

**Monitor:**
```bash
# Watch miner logs
tail -f logs/miner.log

# Look for these patterns:
✅ "Task pulled from validator [X]"
✅ "Generation completed: XXXXX bytes, CLIP=0.XX"
✅ "Task submitted to validator [X]"
✅ "Feedback from [X]: Score=0.XX"
```

### **Step 7: Check Key Metrics**

- [ ] At least 1 task pulled every 2-3 minutes
- [ ] 0 "Generation failed" errors
- [ ] <10% validation failures
- [ ] Positive feedback scores

**Check metrics:**
```bash
# Count tasks in last 15 minutes
grep "Task pulled" logs/miner.log | tail -10

# Check CLIP scores
grep "CLIP Score" logs/generation_service.log | tail -10

# Check feedback
grep "Feedback from" logs/miner.log | tail -10
```

---

## 🎯 Performance Targets

### **Generation Service:**

- [ ] Generation time: 15-25 seconds
- [ ] CLIP scores: >0.7 (>0.6 minimum)
- [ ] File sizes: 100-500KB
- [ ] Success rate: >90%
- [ ] Validation failures: <10%

### **Miner:**

- [ ] Tasks pulled: ~1 per 2-3 minutes
- [ ] Tasks submitted: Match pulled
- [ ] Feedback received: Most submissions
- [ ] No "No available validators" errors
- [ ] No crashes or restarts

### **Expected Throughput:**

- [ ] 15-20 tasks per hour
- [ ] 60-80 tasks per 4 hours (conservative)
- [ ] 100-120 tasks per 4 hours (optimal)

---

## 🔧 Troubleshooting Checklist

### **If Services Won't Start:**

- [ ] Check `logs/generation_service.log` for errors
- [ ] Verify GPU is available (`nvidia-smi`)
- [ ] Check if port 10006 is in use (`netstat -tulpn | grep 10006`)
- [ ] Try stopping and restarting (`./stop_competitive.sh && ./deploy_competitive.sh`)

### **If No Tasks Being Pulled:**

- [ ] Check miner logs for validator skip reasons
- [ ] Verify wallet is registered (`btcli wallet overview`)
- [ ] Check if all validators are blacklisted
- [ ] Verify network connectivity

### **If Validation Failing:**

- [ ] Check CLIP scores in logs
- [ ] Lower validation threshold if needed (`--validation-threshold 0.55`)
- [ ] Increase FLUX steps (`--flux-steps 4` or `8`)
- [ ] Check if CLIP model loaded correctly

### **If Generation Too Slow:**

- [ ] Check GPU utilization (`nvidia-smi`)
- [ ] Use ultra-fast config (`--config configs/text_mv_ultra_fast.yaml`)
- [ ] Reduce FLUX steps (`--flux-steps 2`)
- [ ] Close other GPU applications

---

## 📈 24-Hour Checklist

After deployment, monitor these over 24 hours:

### **Hour 1:**
- [ ] Services still running
- [ ] Tasks being pulled regularly
- [ ] CLIP scores stable >0.7
- [ ] No crashes

### **Hour 4:**
- [ ] Check total tasks completed (~60-80)
- [ ] Review validator feedback
- [ ] Check average generation time
- [ ] Verify no memory leaks

### **Hour 12:**
- [ ] Services still stable
- [ ] Throughput on target (~180-240 tasks)
- [ ] ELO improving (if visible)
- [ ] No unexpected errors

### **Hour 24:**
- [ ] Full day uptime achieved
- [ ] Total tasks ~360-480
- [ ] Consistent quality (CLIP >0.7)
- [ ] Rewards accumulating

---

## ✅ Deployment Success Criteria

Your deployment is successful when ALL of these are true:

- [ ] Both services running without crashes
- [ ] Generation time consistently <25s
- [ ] CLIP scores consistently >0.7
- [ ] Tasks being pulled every 2-3 minutes
- [ ] Tasks being submitted successfully
- [ ] Validator feedback received
- [ ] No critical errors in logs
- [ ] Throughput >60 tasks per 4h
- [ ] Validation failures <10%
- [ ] 24h uptime achieved

---

## 🚨 When to Seek Help

Contact support if:

- [ ] Services crash repeatedly (>3 times)
- [ ] Zero tasks pulled for >30 minutes
- [ ] All validations failing (CLIP <0.6 consistently)
- [ ] Generation time >60s
- [ ] Out of memory errors
- [ ] GPU not detected

**Check these first:**
1. Review logs: `logs/generation_service.log` and `logs/miner.log`
2. Try restarting: `./stop_competitive.sh && ./deploy_competitive.sh`
3. Run tests: `./test_competitive.py`
4. Read troubleshooting: `DEPLOYMENT_GUIDE.md` section 7

---

## 📊 Optimization Checklist

Once stable, consider these optimizations:

### **Speed Optimizations:**

- [ ] Try ultra-fast config if still too slow
- [ ] Adjust FLUX steps (2 vs 4)
- [ ] Lower validation threshold if rejecting too many
- [ ] Increase concurrent workers (if multiple endpoints)

### **Quality Optimizations:**

- [ ] Increase FLUX steps (4 vs 2)
- [ ] Raise validation threshold (0.7 vs 0.6)
- [ ] Use standard fast config (250 iters vs 150)
- [ ] Monitor ELO to verify improvements

### **Throughput Optimizations:**

- [ ] Add more generation endpoints
- [ ] Increase worker count
- [ ] Reduce polling interval
- [ ] Review and update blacklist

---

## 📝 Documentation Review

Make sure you've read:

- [ ] `README_COMPETITIVE.md` - Quick overview
- [ ] `DEPLOYMENT_GUIDE.md` - Full deployment guide
- [ ] `PRODUCTION_CONFIG.md` - Configuration options
- [ ] `COMPETITIVE_IMPLEMENTATION_COMPLETE.md` - Technical details

---

## 🎉 Final Checklist

- [ ] Services deployed successfully
- [ ] All tests passing
- [ ] 15 minutes of stable operation confirmed
- [ ] Monitoring setup and reviewed
- [ ] Performance targets met
- [ ] Documentation read
- [ ] Ready for 24h+ production run

**Congratulations!** Your competitive miner is ready for production! 🚀

---

## 📞 Quick Reference

**Deploy:**
```bash
./deploy_competitive.sh
```

**Monitor:**
```bash
tail -f logs/miner.log
```

**Test:**
```bash
./test_competitive.py
```

**Stop:**
```bash
./stop_competitive.sh
```

**Health:**
```bash
curl http://localhost:10006/health
```

**Restart:**
```bash
./stop_competitive.sh && ./deploy_competitive.sh
```

---

**Deployment Date**: ________________

**Initial Metrics**:
- ELO: _______
- Avg CLIP: _______
- Tasks/4h: _______

**Post-Deploy Metrics (24h)**:
- ELO: _______
- Avg CLIP: _______
- Tasks/4h: _______

---

**Good luck mining!** 🚀
