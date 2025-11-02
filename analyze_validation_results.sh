#!/bin/bash

echo "=========================================="
echo "DETAILED VALIDATION ANALYSIS"
echo "=========================================="
echo ""

# File size distribution
echo "=== File Size Distribution ==="
ls -lh /tmp/validation_test_*.ply 2>/dev/null | awk '{
  if($5 ~ /K/) size=substr($5,1,length($5)-1)/1024
  else if($5 ~ /M/) size=substr($5,1,length($5)-1)
  else size=0

  if(size > 0) {
    sizes[NR]=size
    sum+=size
    count++

    if(size < 1) tiny++
    else if(size < 10) small++
    else if(size < 20) medium++
    else if(size < 30) large++
    else huge++
  }
}
END {
  print "Generated " count " files"
  print "Average size: " sprintf("%.1f", sum/count) " MB"
  print ""
  print "Distribution:"
  print "  < 1 MB:  " tiny " files (FAILED - too small)"
  print "  1-10 MB: " small " files (marginal quality)"
  print "  10-20 MB:" medium " files (good quality)"
  print "  20-30 MB:" large " files (excellent quality)"
  print "  > 30 MB: " huge " files (exceptional quality)"
}'

echo ""
echo "=== Gaussian Count Distribution ==="
# Extract gaussian counts from logs during test period
tail -1000 ~/.pm2/logs/gen-worker-1-error.log | \
  grep -oP "Gaussians: \K[0-9,]+" | \
  tr -d ',' | \
  tail -30 | \
  sort -n | \
  awk '{
    counts[NR]=$1
    sum+=$1

    if($1 < 100000) under100k++
    else if($1 < 150000) under150k++
    else if($1 < 200000) under200k++
    else if($1 < 250000) under250k++
    else if($1 < 300000) under300k++
    else over300k++
  }
  END {
    if(NR > 0) {
      print "Samples analyzed: " NR
      print "Min:     " counts[1]
      print "Median:  " counts[int(NR/2)]
      print "Max:     " counts[NR]
      print "Average: " int(sum/NR)
      print ""
      print "Distribution:"
      print "  < 100K:   " under100k " (" int(under100k*100/NR) "%) VERY SPARSE"
      print "  100-150K: " under150k " (" int(under150k*100/NR) "%) SPARSE - REJECTED"
      print "  150-200K: " under200k " (" int(under200k*100/NR) "%) MARGINAL"
      print "  200-250K: " under250k " (" int(under250k*100/NR) "%) GOOD"
      print "  250-300K: " under300k " (" int(under300k*100/NR) "%) VERY GOOD"
      print "  > 300K:   " over300k " (" int(over300k*100/NR) "%) EXCELLENT"
    }
  }'

echo ""
echo "=== Retry Logic Performance ==="
# Check if retry logic was triggered and how effective it was
retries=$(tail -1000 ~/.pm2/logs/gen-worker-1-error.log | grep "🔄 Retry result" | wc -l)
retry_success=$(tail -1000 ~/.pm2/logs/gen-worker-1-error.log | grep "✅ RETRY SUCCESSFUL" | wc -l)
retry_improved=$(tail -1000 ~/.pm2/logs/gen-worker-1-error.log | grep "⚠️.*RETRY IMPROVED" | wc -l)
retry_failed=$(tail -1000 ~/.pm2/logs/gen-worker-1-error.log | grep "❌ RETRY FAILED" | wc -l)

echo "Retry attempts: $retries"
echo "  Successful retries: $retry_success (fully passed threshold)"
echo "  Improved retries:   $retry_improved (better but still sparse)"
echo "  Failed retries:     $retry_failed (no improvement)"

if [ $retries -gt 0 ]; then
  effectiveness=$(echo "scale=1; ($retry_success + $retry_improved) * 100 / $retries" | bc)
  echo "  Retry effectiveness: ${effectiveness}%"
fi

echo ""
echo "=== Quality Gate Statistics ==="
sparse_detected=$(tail -1000 ~/.pm2/logs/gen-worker-1-error.log | grep "⚠️.*SPARSE GENERATION" | wc -l)
final_failures=$(tail -1000 ~/.pm2/logs/gen-worker-1-error.log | grep "❌ QUALITY GATE FAILED" | wc -l)

echo "Sparse generations detected: $sparse_detected"
echo "Final quality gate failures: $final_failures (after retry attempts)"
echo "Retry recovery rate: $((sparse_detected - final_failures)) recovered from $sparse_detected sparse"

echo ""
echo "=========================================="
echo "OVERALL ASSESSMENT"
echo "=========================================="

# Calculate from file sizes (most reliable metric)
total_files=$(ls /tmp/validation_test_*.ply 2>/dev/null | wc -l)
successful_files=$(ls -lh /tmp/validation_test_*.ply 2>/dev/null | awk '$5 ~ /M/ && $5 !~ /^0/' | wc -l)

if [ $total_files -gt 0 ]; then
  success_rate=$(echo "scale=1; $successful_files * 100 / $total_files" | bc)

  echo "Final Success Rate: ${success_rate}%"
  echo ""

  if (( $(echo "$success_rate >= 75" | bc -l) )); then
    echo "✅ RECOMMENDATION: GO TO MAINNET"
    echo "   Success rate is excellent. Ready for production."
  elif (( $(echo "$success_rate >= 60" | bc -l) )); then
    echo "⚠️  RECOMMENDATION: PROCEED WITH CAUTION"
    echo "   Success rate is acceptable but monitor closely."
    echo "   Consider testnet first if available."
  elif (( $(echo "$success_rate >= 50" | bc -l) )); then
    echo "❌ RECOMMENDATION: IMPROVE BEFORE MAINNET"
    echo "   Success rate too low. Increase enhancement parameters."
    echo "   Suggested: 4.0x sharpness, 2.0x contrast baseline"
  else
    echo "❌ RECOMMENDATION: DO NOT GO TO MAINNET"
    echo "   Success rate critically low. Further investigation needed."
    echo "   Check TRELLIS microservice stability and logs."
  fi
fi
