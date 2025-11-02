#!/bin/bash
# Real-time critical error monitoring (runs in parallel with main monitor)

echo "🚨 CRITICAL ERROR MONITORING STARTED: $(date)"
echo "Watching for: exceptions, crashes, OOM, CUDA errors"
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Monitor all three services in parallel
(tail -f ~/.pm2/logs/gen-worker-1-error.log | grep --line-buffered -i "exception\|traceback\|out of memory\|cuda\|RuntimeError\|ValueError" | while read line; do
    echo "[WORKER ERROR] $(date +%H:%M:%S): $line"
done) &

(tail -f ~/.pm2/logs/miner-sn17-mainnet-error.log | grep --line-buffered -i "exception\|traceback\|connection\|timeout" | while read line; do
    echo "[MINER ERROR] $(date +%H:%M:%S): $line"
done) &

(tail -f ~/.pm2/logs/trellis-microservice-error.log | grep --line-buffered -i "error\|exception\|failed" | while read line; do
    echo "[TRELLIS ERROR] $(date +%H:%M:%S): $line"
done) &

# Wait for 1 hour
sleep 3600

# Kill background processes
pkill -P $$

echo ""
echo "=========================================="
echo "🚨 CRITICAL ERROR MONITORING COMPLETE: $(date)"
