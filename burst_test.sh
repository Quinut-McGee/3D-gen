#!/bin/bash

echo "=========================================="
echo "BURST LOAD TEST"
echo "=========================================="
echo "Sending 10 rapid-fire requests (simulating validator burst)..."
echo "Start time: $(date)"
echo ""

# Record initial state
initial_restarts=$(pm2 describe trellis-microservice 2>/dev/null | grep 'restarts' | awk '{print $3}')

# Send 10 requests as fast as possible (no delays)
for i in {1..10}; do
  (
    echo "Request $i started at $(date +%s.%N)"
    timeout 90 curl -s -X POST http://localhost:10010/generate/ \
      -F "prompt=burst test $i" \
      -o /tmp/burst_test_$i.glb \
      -w "%{http_code}\n" 2>&1
    echo "Request $i completed at $(date +%s.%N)"
  ) &
done

# Wait for all requests to complete
wait

echo ""
echo "All requests completed. Checking results..."
echo ""

# Check success rate
successful=0
for i in {1..10}; do
  size=$(stat -c%s /tmp/burst_test_$i.glb 2>/dev/null || echo "0")
  if [ "$size" -gt 1048576 ]; then
    echo "  Request $i: ✅ Success ($(echo "scale=1; $size / 1048576" | bc)MB)"
    ((successful++))
  else
    echo "  Request $i: ❌ Failed (size: $size bytes)"
  fi
done

# Check if TRELLIS crashed
final_restarts=$(pm2 describe trellis-microservice 2>/dev/null | grep 'restarts' | awk '{print $3}')

echo ""
echo "Results:"
echo "  Successful: $successful/10"
echo "  TRELLIS Restarts: $initial_restarts → $final_restarts"
echo ""

if [ "$final_restarts" -gt "$initial_restarts" ]; then
  echo "🚨 WARNING: TRELLIS RESTARTED DURING BURST TEST!"
  exit 1
else
  echo "✅ TRELLIS remained stable during burst"
fi
