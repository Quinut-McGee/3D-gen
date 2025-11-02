#!/bin/bash
# Comprehensive monitoring script for miner services (1 hour)
# Monitors: gen-worker-1, miner-sn17-mainnet, trellis-microservice

LOG_FILE="/home/kobe/404-gen/v1/3D-gen/generation/monitoring_$(date +%Y%m%d_%H%M%S).log"
DURATION=3600  # 1 hour in seconds
START_TIME=$(date +%s)
CHECK_INTERVAL=60  # Check every 60 seconds

echo "🚀 MONITORING STARTED: $(date)" | tee -a "$LOG_FILE"
echo "Duration: 1 hour (3600s)" | tee -a "$LOG_FILE"
echo "Check interval: 60s" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Counters for statistics
TOTAL_CHECKS=0
ERRORS_FOUND=0
RESTARTS_DETECTED=0
GOOD_SCORES=0
BAD_SCORES=0

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    # Check if 1 hour has passed
    if [ $ELAPSED -ge $DURATION ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        echo "⏰ MONITORING COMPLETE: $(date)" | tee -a "$LOG_FILE"
        echo "Duration: $ELAPSED seconds" | tee -a "$LOG_FILE"
        echo "Total checks: $TOTAL_CHECKS" | tee -a "$LOG_FILE"
        echo "Errors found: $ERRORS_FOUND" | tee -a "$LOG_FILE"
        echo "Restarts detected: $RESTARTS_DETECTED" | tee -a "$LOG_FILE"
        echo "Good scores: $GOOD_SCORES" | tee -a "$LOG_FILE"
        echo "Bad scores (0.0): $BAD_SCORES" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        break
    fi

    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    REMAINING=$((DURATION - ELAPSED))

    echo "[CHECK #$TOTAL_CHECKS] $(date) - Remaining: ${REMAINING}s" | tee -a "$LOG_FILE"

    # 1. Check PM2 service status
    echo "  📊 Service Status:" | tee -a "$LOG_FILE"
    pm2 status | grep -E "gen-worker-1|miner-sn17-mainnet|trellis-microservice" | tee -a "$LOG_FILE"

    # Check for crashed services
    CRASHED=$(pm2 status | grep -E "gen-worker-1|miner-sn17-mainnet|trellis-microservice" | grep -c "stopped\|errored")
    if [ $CRASHED -gt 0 ]; then
        echo "  ⚠️  WARNING: $CRASHED service(s) crashed!" | tee -a "$LOG_FILE"
        ERRORS_FOUND=$((ERRORS_FOUND + 1))
        echo "  🔄 Attempting restart..." | tee -a "$LOG_FILE"
        pm2 restart all | tee -a "$LOG_FILE"
        RESTARTS_DETECTED=$((RESTARTS_DETECTED + 1))
        sleep 10
    fi

    # 2. Check for errors in logs (last 60 seconds)
    echo "  🔍 Checking for errors..." | tee -a "$LOG_FILE"

    # Worker errors
    WORKER_ERRORS=$(tail -100 ~/.pm2/logs/gen-worker-1-error.log | grep -i "error\|exception\|failed\|critical" | grep -v "ERROR" | tail -3)
    if [ ! -z "$WORKER_ERRORS" ]; then
        echo "  ⚠️  gen-worker-1 errors:" | tee -a "$LOG_FILE"
        echo "$WORKER_ERRORS" | tee -a "$LOG_FILE"
        ERRORS_FOUND=$((ERRORS_FOUND + 1))
    fi

    # Miner errors
    MINER_ERRORS=$(tail -100 ~/.pm2/logs/miner-sn17-mainnet-error.log | grep -i "error\|exception\|failed\|critical" | tail -3)
    if [ ! -z "$MINER_ERRORS" ]; then
        echo "  ⚠️  miner errors:" | tee -a "$LOG_FILE"
        echo "$MINER_ERRORS" | tee -a "$LOG_FILE"
        ERRORS_FOUND=$((ERRORS_FOUND + 1))
    fi

    # TRELLIS errors
    TRELLIS_ERRORS=$(tail -100 ~/.pm2/logs/trellis-microservice-error.log | grep -i "error\|exception\|failed" | tail -3)
    if [ ! -z "$TRELLIS_ERRORS" ]; then
        echo "  ⚠️  TRELLIS errors:" | tee -a "$LOG_FILE"
        echo "$TRELLIS_ERRORS" | tee -a "$LOG_FILE"
        ERRORS_FOUND=$((ERRORS_FOUND + 1))
    fi

    # 3. Check recent validator scores
    echo "  💰 Recent validator scores (last 60s):" | tee -a "$LOG_FILE"
    RECENT_SCORES=$(tail -100 ~/.pm2/logs/miner-sn17-mainnet-out.log | grep "Score=" | tail -3)
    if [ ! -z "$RECENT_SCORES" ]; then
        echo "$RECENT_SCORES" | tee -a "$LOG_FILE"

        # Count good vs bad scores
        ZERO_SCORES=$(echo "$RECENT_SCORES" | grep -c "Score=0.0")
        NONZERO_SCORES=$(echo "$RECENT_SCORES" | grep "Score=" | grep -v "Score=0.0" | wc -l)

        GOOD_SCORES=$((GOOD_SCORES + NONZERO_SCORES))
        BAD_SCORES=$((BAD_SCORES + ZERO_SCORES))

        if [ $ZERO_SCORES -gt 0 ]; then
            echo "  ⚠️  $ZERO_SCORES Score=0.0 detected in last check" | tee -a "$LOG_FILE"
        fi
    fi

    # 4. Check memory usage
    echo "  💾 Memory usage:" | tee -a "$LOG_FILE"
    pm2 status | grep -E "gen-worker-1|miner-sn17-mainnet|trellis-microservice" | awk '{print "    " $2 ": " $11}' | tee -a "$LOG_FILE"

    # Check for high memory (>40GB for worker)
    WORKER_MEM=$(pm2 jlist | jq -r '.[] | select(.name=="gen-worker-1") | .monit.memory' | awk '{print int($1/1024/1024/1024)}')
    if [ ! -z "$WORKER_MEM" ] && [ $WORKER_MEM -gt 40 ]; then
        echo "  ⚠️  WARNING: gen-worker-1 memory usage high: ${WORKER_MEM}GB" | tee -a "$LOG_FILE"
        ERRORS_FOUND=$((ERRORS_FOUND + 1))
    fi

    # 5. Check GPU memory
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    GPU_PERCENT=$((GPU_MEM * 100 / GPU_TOTAL))
    echo "  🎮 GPU memory: ${GPU_MEM}MB / ${GPU_TOTAL}MB (${GPU_PERCENT}%)" | tee -a "$LOG_FILE"

    if [ $GPU_PERCENT -gt 95 ]; then
        echo "  ⚠️  WARNING: GPU memory usage critical: ${GPU_PERCENT}%" | tee -a "$LOG_FILE"
        ERRORS_FOUND=$((ERRORS_FOUND + 1))
    fi

    # 6. Check for timeout violations
    TIMEOUT_VIOLATIONS=$(tail -100 ~/.pm2/logs/gen-worker-1-error.log | grep -c "⏱️\|Timing safety")
    if [ $TIMEOUT_VIOLATIONS -gt 0 ]; then
        echo "  ⏱️  Timeout safety check triggered: $TIMEOUT_VIOLATIONS times" | tee -a "$LOG_FILE"
    fi

    echo "" | tee -a "$LOG_FILE"

    # Sleep until next check
    sleep $CHECK_INTERVAL
done

echo "" | tee -a "$LOG_FILE"
echo "Monitoring log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
