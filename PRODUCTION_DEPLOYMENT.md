# 🚀 Production Deployment - Step by Step

Follow these steps exactly to deploy your competitive miner.

---

## 📍 Current Location

You are in: `/root/github/3D-gen/generation`

---

## Step 1: Install New Dependencies for Competitive System

```bash
# You're already in generation/, stay here
cd /root/github/3D-gen/generation

# Activate the mining environment
conda activate three-gen-mining

# Install new packages for competitive system
pip install transformers diffusers accelerate xformers
pip install git+https://github.com/openai/CLIP.git

# Verify installation
python -c "import transformers; import diffusers; import clip; print('✅ All packages installed')"
```

---

## Step 2: Update Generation Service Config

```bash
# Create new PM2 config for competitive generation
cat > generation.competitive.config.js << 'EOF'
module.exports = {
  apps : [{
    name: 'generation-competitive',
    script: 'serve_competitive.py',
    interpreter: '/root/miniconda3/envs/three-gen-mining/bin/python',
    args: '--port 8093 --config configs/text_mv_fast.yaml --flux-steps 4 --validation-threshold 0.6 --enable-validation'
  }]
};
EOF

# Verify config created
cat generation.competitive.config.js
```

---

## Step 3: Stop Old Generation Service

```bash
# Stop old service
pm2 stop generation

# Or if you want to delete it
pm2 delete generation

# Verify stopped
pm2 list
```

---

## Step 4: Start Competitive Generation Service

```bash
# Start new competitive service
pm2 start generation.competitive.config.js

# Watch logs (this will take 2-3 minutes for model loading)
pm2 logs generation-competitive --lines 50
```

**Wait for this output:**
```
[1/4] Loading FLUX.1-schnell (text-to-image)...
✅ FLUX.1-schnell ready

[2/4] Loading BRIA RMBG 2.0 (background removal)...
✅ BRIA RMBG 2.0 ready

[3/4] Loading DreamGaussian (3D generation)...
✅ DreamGaussian ready

[4/4] Loading CLIP validator...
✅ CLIP validator ready (threshold=0.6)

🚀 COMPETITIVE MINER READY FOR PRODUCTION
```

**This will take 2-3 minutes.** Press `Ctrl+C` when you see "READY FOR PRODUCTION".

---

## Step 5: Test Generation Service

```bash
# Test generation (open new terminal, or press Ctrl+C to exit logs)
curl -X POST http://127.0.0.1:8093/generate/ \
     -F "prompt=a red cube" \
     -o /tmp/test_competitive.ply

# Check output
ls -lh /tmp/test_competitive.ply   # Should be 100-500KB
head /tmp/test_competitive.ply     # Should start with "ply"

# Check time (should be ~20 seconds)
time curl -X POST http://127.0.0.1:8093/generate/ \
     -F "prompt=a blue sphere" \
     -o /tmp/test2.ply
```

**Expected:**
- File size: 100-500KB ✅
- Time: 15-25 seconds ✅
- Starts with "ply" ✅

---

## Step 6: Install Neuron Dependencies

```bash
# Go to neurons directory
cd /root/github/3D-gen/neurons

# Activate neurons environment
conda activate three-gen-neurons

# Install async dependencies (aiohttp should already be installed, but verify)
pip install aiohttp
python -c "import aiohttp; print('✅ aiohttp installed')"
```

---

## Step 7: Update Miner Config

```bash
# Still in neurons/ directory
cat > miner.competitive.config.js << 'EOF'
module.exports = {
  apps : [{
    name: 'miner-competitive',
    script: 'serve_miner_competitive.py',
    interpreter: '/root/miniconda3/envs/three-gen-neurons/bin/python',
    args: '--wallet.name validator --wallet.hotkey sn17miner2 --subtensor.network finney --netuid 17 --generation.endpoint http://127.0.0.1:8093 --logging.debug'
  }]
};
EOF

# Verify config created
cat miner.competitive.config.js
```

---

## Step 8: Stop Old Miner

```bash
# Stop old miner
pm2 stop miner

# Or delete it
pm2 delete miner

# Verify stopped
pm2 list
```

---

## Step 9: Start Competitive Miner

```bash
# Start new competitive miner
pm2 start miner.competitive.config.js

# Watch logs
pm2 logs miner-competitive --lines 100
```

**Expected output:**
```
=============================================================
🚀 COMPETITIVE MINER STARTING
=============================================================
UID: XX
Workers: 1
Endpoints: ['http://127.0.0.1:8093']
Validator polling: every 10.0s
=============================================================
```

---

## Step 10: Monitor Initial Operation (15 minutes)

```bash
# Watch miner logs in real-time
pm2 logs miner-competitive

# Look for:
✅ "Task pulled from validator [X]"
✅ "Processing task 'prompt' from validator X"
✅ "Generation completed: XXXXX bytes, CLIP=0.XX"
✅ "Task submitted to validator [X]"
✅ "Feedback from [X]: Score=0.XX, Reward=0.XXX"
```

**What you should see every 2-3 minutes:**
1. Task pulled from validator
2. Generation request sent to service
3. Generation completed (with CLIP score)
4. Task submitted
5. Feedback received

---

## Step 11: Save PM2 Configuration

```bash
# Save current PM2 state
pm2 save

# Set PM2 to start on reboot
pm2 startup
# Follow the instructions it gives you
```

---

## 🎯 Success Criteria

After 15 minutes, verify:

- [ ] Both services running: `pm2 list`
- [ ] Generation service shows uptime >15min
- [ ] Miner shows uptime >15min
- [ ] At least 5 tasks pulled and submitted
- [ ] CLIP scores >0.7 in generation logs
- [ ] Feedback received from validators
- [ ] No errors in logs

**Check:**
```bash
# Services status
pm2 list

# Count tasks
pm2 logs miner-competitive --nostream | grep "Task pulled" | wc -l
pm2 logs miner-competitive --nostream | grep "Feedback from" | wc -l

# Check CLIP scores
pm2 logs generation-competitive --nostream | grep "CLIP Score" | tail -10

# Check for errors
pm2 logs miner-competitive --err --lines 50
pm2 logs generation-competitive --err --lines 50
```

---

## 🔧 Troubleshooting

### Issue: Generation service won't start

```bash
# Check detailed logs
pm2 logs generation-competitive --err --lines 100

# Common issues:
# - Out of VRAM: Close other GPU apps, or reduce flux-steps to 2
# - Missing dependencies: Re-run pip install commands from Step 1
# - Port in use: Change --port 8093 to --port 8094
```

### Issue: Miner shows "No available validators"

```bash
# Check miner logs for skip reasons
pm2 logs miner-competitive | grep "skipped"

# Common fixes:
# - Wait a few minutes, validators might be in cooldown
# - Check you're registered: btcli wallet overview
# - Verify network connection
```

### Issue: All validations failing (CLIP <0.6)

```bash
# Lower threshold temporarily
pm2 delete generation-competitive

# Edit config to lower threshold
cat > generation.competitive.config.js << 'EOF'
module.exports = {
  apps : [{
    name: 'generation-competitive',
    script: 'serve_competitive.py',
    interpreter: '/root/miniconda3/envs/three-gen-mining/bin/python',
    args: '--port 8093 --config configs/text_mv_fast.yaml --flux-steps 4 --validation-threshold 0.55 --enable-validation'
  }]
};
EOF

pm2 start generation.competitive.config.js
```

### Issue: Generation too slow (>30s)

```bash
# Use ultra-fast config
pm2 delete generation-competitive

cat > generation.competitive.config.js << 'EOF'
module.exports = {
  apps : [{
    name: 'generation-competitive',
    script: 'serve_competitive.py',
    interpreter: '/root/miniconda3/envs/three-gen-mining/bin/python',
    args: '--port 8093 --config configs/text_mv_ultra_fast.yaml --flux-steps 2 --validation-threshold 0.6 --enable-validation'
  }]
};
EOF

pm2 start generation.competitive.config.js
```

---

## 📊 Monitoring Commands

```bash
# View all services
pm2 list

# Watch miner logs
pm2 logs miner-competitive

# Watch generation logs
pm2 logs generation-competitive

# Watch both
pm2 logs

# Check GPU usage
nvidia-smi

# Restart services
pm2 restart miner-competitive
pm2 restart generation-competitive

# Stop everything
pm2 stop all

# Start everything
pm2 start all
```

---

## 🔄 Rollback (if needed)

If something goes wrong and you need to go back to the old system:

```bash
# Stop competitive services
pm2 stop miner-competitive generation-competitive
pm2 delete miner-competitive generation-competitive

# Start old services
cd /root/github/3D-gen/generation
pm2 start generation.config.js

cd /root/github/3D-gen/neurons
pm2 start miner.config.js

# Verify
pm2 list
```

---

## ✅ Next Steps

Once everything is stable for 4 hours:

1. **Check throughput:** Should be 60-80 tasks per 4h (vs ~10 before)
2. **Check CLIP scores:** Should average >0.7
3. **Check ELO:** Should start improving
4. **Check rewards:** Should be 10-15x higher

---

## 📞 Quick Reference

**Your setup:**
- Wallet: `validator`
- Hotkey: `sn17miner2`
- Generation port: `8093`
- Generation endpoint: `http://127.0.0.1:8093`
- Network: `finney`
- Subnet: `17`

**PM2 process names:**
- Generation: `generation-competitive`
- Miner: `miner-competitive`

**Config files:**
- Generation: `/root/github/3D-gen/generation/generation.competitive.config.js`
- Miner: `/root/github/3D-gen/neurons/miner.competitive.config.js`

---

## 🎉 You're Ready!

Follow the steps above and your competitive miner will be running with:
- ✅ FLUX.1-schnell (3s text-to-image)
- ✅ BRIA RMBG 2.0 (0.2s background removal)
- ✅ DreamGaussian (15s 3D generation)
- ✅ CLIP validation (prevents penalties)
- ✅ Async multi-validator polling
- ✅ Expected 10-15x improvement

**Start with Step 1 and work through each step carefully.** 🚀
