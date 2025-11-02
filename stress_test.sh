#!/bin/bash

echo "=========================================="
echo "INFRASTRUCTURE STRESS TEST"
echo "=========================================="
echo "Testing sustained load for 30 minutes..."
echo "Start time: $(date)"
echo ""

# Monitor script running in background
(
  while true; do
    echo "=== $(date) ==="
    echo "TRELLIS Status: $(pm2 describe trellis-microservice 2>/dev/null | grep 'status' | awk '{print $3}')"
    echo "TRELLIS Restarts: $(pm2 describe trellis-microservice 2>/dev/null | grep 'restarts' | awk '{print $3}')"
    echo "Worker Status: $(pm2 describe gen-worker-1 2>/dev/null | grep 'status' | awk '{print $3}')"
    echo "GPU Memory: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) MB"
    echo ""
    sleep 60
  done
) > stress_test_monitor.log 2>&1 &
monitor_pid=$!

# Generate continuous load (60 generations over 30 minutes)
successful=0
failed=0

for i in {1..60}; do
  echo "[$i/60] Generation attempt at $(date)"

  # Generate with timeout
  timeout 90 curl -s -X POST http://localhost:10010/generate/ \
    -F "prompt=stress test object $i" \
    -o /tmp/stress_test_$i.glb \
    2>&1 | tail -2

  # Check if successful
  size=$(stat -c%s /tmp/stress_test_$i.glb 2>/dev/null || echo "0")

  if [ "$size" -gt 1048576 ]; then
    echo "  ✅ Success ($(echo "scale=1; $size / 1048576" | bc)MB)"
    ((successful++))
  else
    echo "  ❌ Failed (size: $size bytes)"
    ((failed++))

    # Check if TRELLIS crashed
    trellis_status=$(pm2 describe trellis-microservice 2>/dev/null | grep 'status' | awk '{print $3}')
    if [ "$trellis_status" != "online" ]; then
      echo "  🚨 TRELLIS CRASHED!"
      kill $monitor_pid
      exit 1
    fi
  fi

  # Brief pause between generations (30s per generation)
  sleep 25
done

# Stop monitoring
kill $monitor_pid

echo ""
echo "=========================================="
echo "STRESS TEST COMPLETE"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  Successful: $successful/60"
echo "  Failed: $failed/60"
echo "  Success Rate: $(echo "scale=1; $successful * 100 / 60" | bc)%"
echo ""
echo "TRELLIS Final Status:"
pm2 describe trellis-microservice | grep -E "status|restarts|uptime"
