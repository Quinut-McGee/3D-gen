# Infrastructure Stability Audit
**Date:** November 2, 2025
**Audit Type:** Pre-Mainnet Deployment Verification

## Executive Summary

🚨 **CRITICAL BLOCKER FOUND**: Wallet not registered on mainnet (netuid 17)

### Current System Status
- **gen-worker-1**: ✅ Online, 10h uptime, 4 restarts, 32.3GB RAM
- **trellis-microservice**: ✅ Online, 10h uptime, 2 restarts, 6.2GB RAM  
- **miner-sn17-mainnet**: ❌ CRASH LOOP - 16,660 restarts, wallet not registered

---

## Phase 1: Historical Issues Analysis

### 1.A TRELLIS Microservice Crashes

#### Finding: No Hard Crashes Detected
- **TRELLIS restart count**: 2 (over 10h period)
- **Unstable restarts**: 0
- **CUDA/OOM errors**: None found in logs
- **Python exceptions**: None found in logs

✅ **Assessment**: TRELLIS microservice is stable under current load

#### Root Cause of Previous Issues
Based on log analysis, the TRELLIS issues were related to:
1. **Startup delays**: TRELLIS takes time to load the model (~90s)
2. **Connection timing**: Worker tried connecting before TRELLIS was ready
3. **No hard crashes**: System was restarting cleanly, not crashing

### 1.B Worker Connection Issues

#### Finding: 14 Connection Failures During Startup Periods

**Timeline of Connection Failures:**
- Nov 1, 2025: 12 failures between 03:51 - 04:12 (21-minute window)
- Nov 2, 2025: 2 failures between 03:31 - 03:36 (5-minute window)

**Pattern Analysis:**
- All failures are "Connection refused" errors
- Failures occur during TRELLIS startup/restart periods
- No timeout errors during actual generation
- Worker gracefully handles connection failures with retry logic

✅ **Assessment**: Connection failures are expected during startup, handled properly

#### Generation Timeouts
- **Count**: 0
- **Assessment**: No timeout issues once TRELLIS is running

### 1.C CRITICAL BLOCKER: Wallet Not Registered

#### Finding: Miner Cannot Connect to Mainnet

**Error:**
```
RuntimeError: Wallet Wallet (Name: 'validator', Hotkey: 'sn17miner2', Path: '~/.bittensor/wallets/') 
not registered on netuid 17. Please register using `btcli subnets register`
```

**Impact:**
- Miner has crashed 16,660 times trying to connect
- Cannot mine on subnet 17 without wallet registration
- **This is a complete blocker for mainnet deployment**

❌ **BLOCKER**: Must register wallet before any mainnet deployment

---

## Fixes Applied and Verification Status

### Fix 1: CUDA Memory Management
**Implemented**: Yes (generation/trellis_integration.py)
- Added `torch.cuda.empty_cache()` after each generation
- TRELLIS pipeline stays loaded (not reloading each time)
**Verification**: ✅ No OOM errors in recent logs

### Fix 2: Connection Retry Logic
**Implemented**: Yes (gen-worker-1 handles connection failures)
- Worker retries connection to TRELLIS
- Graceful handling of "Connection refused"
**Verification**: ✅ Worker recovers from TRELLIS restarts

### Fix 3: Timeout Protection
**Status**: Needs verification in Phase 2C
**Expected**: 60s timeout on API calls to TRELLIS

### Fix 4: Wallet Registration
**Status**: ❌ NOT FIXED - blocking issue
**Required Action**: Register 'sn17miner2' wallet on netuid 17

---

## Issues Summary

| Issue | Severity | Status | Action Required |
|-------|----------|--------|-----------------|
| Wallet not registered | 🔴 CRITICAL | ❌ BLOCKER | Register wallet with `btcli subnets register` |
| TRELLIS startup delays | 🟡 MEDIUM | ✅ HANDLED | None - retry logic works |
| Connection failures during startup | 🟡 MEDIUM | ✅ HANDLED | None - expected behavior |
| CUDA memory management | 🟢 LOW | ✅ FIXED | None - working correctly |

---

## Next Steps

**BEFORE ANY MAINNET DEPLOYMENT:**

1. ⚠️ **CRITICAL**: Register the sn17miner2 wallet on netuid 17
   ```bash
   btcli subnets register --netuid 17 --wallet.name validator --wallet.hotkey sn17miner2
   ```

2. Verify wallet registration successful

3. Stop the crash-looping miner:
   ```bash
   pm2 stop miner-sn17-mainnet
   ```

4. Complete infrastructure validation (Phases 2-4)

5. Only proceed to mainnet if all tests pass

---

## Phase 1 Conclusion

**TRELLIS Infrastructure**: ✅ STABLE  
**Worker Infrastructure**: ✅ STABLE  
**Mainnet Connection**: ❌ BLOCKED (wallet not registered)

**Recommendation**: DO NOT DEPLOY until wallet is registered and validated.

