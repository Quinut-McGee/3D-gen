#!/bin/bash

echo "=========================================="
echo "TRELLIS VALIDATION TEST - 30 Generations"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Clean up old test files
rm -f /tmp/validation_test_*.ply 2>/dev/null

# Array of diverse test prompts (mix of simple and complex)
prompts=(
  "simple wooden cube"
  "smooth metal sphere"
  "clear glass bottle"
  "rough stone pillar"
  "red sports car"
  "blue office chair"
  "green dining table"
  "yellow desk lamp"
  "ornate golden crown"
  "simple silver ring"
  "rustic wooden barrel"
  "modern ceramic vase"
  "ancient stone statue"
  "sleek smartphone"
  "vintage film camera"
  "polished marble column"
  "weathered brick wall section"
  "shiny copper teapot"
  "frosted glass sphere"
  "carved wooden box"
  "metal industrial pipe"
  "ceramic coffee mug"
  "leather armchair"
  "chrome bicycle handlebar"
  "woven wicker basket"
  "painted wooden toy"
  "stainless steel pot"
  "terra cotta flower pot"
  "bronze small statue"
  "plastic storage container"
)

successful=0
failed=0
total=30

# Run 30 generations
for i in $(seq 0 29); do
  prompt="${prompts[$i]}"

  echo "----------------------------------------"
  echo "Test $((i+1))/30: '$prompt'"
  echo "----------------------------------------"

  # Generate with timeout
  timeout 90 curl -X POST http://localhost:10010/generate/ \
    -F "prompt=$prompt" \
    -o /tmp/validation_test_$i.ply \
    -w "\n  HTTP Status: %{http_code}\n  Total time: %{time_total}s\n  Size: %{size_download} bytes\n" \
    2>&1 | tail -4

  # Check if generation succeeded (file size > 1MB = ~150K gaussians minimum)
  size=$(stat -c%s /tmp/validation_test_$i.ply 2>/dev/null || echo "0")
  size_mb=$(echo "scale=1; $size / 1048576" | bc)

  if [ "$size" -gt 1048576 ]; then
    echo "  ✅ SUCCESS: ${size_mb}MB"
    ((successful++))
  else
    echo "  ❌ FAILED: ${size_mb}MB (too small)"
    ((failed++))
  fi

  # Brief pause between generations to avoid overload
  sleep 5
done

echo ""
echo "=========================================="
echo "TEST COMPLETE"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  Total: $total"
echo "  Successful: $successful"
echo "  Failed: $failed"
echo "  Success Rate: $(echo "scale=1; $successful * 100 / $total" | bc)%"
echo ""
echo "Analyzing logs for detailed metrics..."
