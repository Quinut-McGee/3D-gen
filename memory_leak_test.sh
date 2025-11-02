#!/bin/bash

echo "=========================================="
echo "MEMORY LEAK TEST"
echo "=========================================="
echo "Monitoring memory usage over 20 generations..."
echo ""

for i in {1..20}; do
  echo "Generation $i/20:"

  # Record memory before
  mem_before=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "  GPU Memory before: ${mem_before}MB"

  # Generate
  timeout 90 curl -s -X POST http://localhost:10010/generate/ \
    -F "prompt=memory test $i" \
    -o /tmp/memory_test_$i.glb 2>&1 > /dev/null

  # Record memory after
  sleep 5  # Let cleanup happen
  mem_after=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "  GPU Memory after:  ${mem_after}MB"

  # Check if memory is growing
  delta=$((mem_after - mem_before))
  if [ $delta -gt 500 ]; then
    echo "  ⚠️  WARNING: Memory increased by ${delta}MB"
  else
    echo "  ✅ Memory stable (delta: ${delta}MB)"
  fi

  echo ""
done

echo "Memory leak test complete. Check if memory grew unbounded."
