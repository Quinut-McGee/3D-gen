#!/bin/bash

echo "=========================================="
echo "OPTIMIZATION INSIGHTS ANALYSIS"
echo "=========================================="
echo ""

# Extract all gaussian counts and file sizes from validation
echo "=== Detailed Generation Analysis ==="
echo ""

# Get generation times and sizes from validation test
grep "Total time:" validation_test_output.txt | awk '{print $3}' | sed 's/s$//' > /tmp/times.txt
grep "Size:" validation_test_output.txt | awk '{print $2}' > /tmp/sizes.txt

# Calculate time statistics
echo "Generation Time Analysis:"
awk '{sum+=$1; sumsq+=$1*$1; if(NR==1){min=$1;max=$1} if($1<min){min=$1} if($1>max){max=$1}} 
END {
  avg=sum/NR; 
  stddev=sqrt(sumsq/NR - avg*avg);
  print "  Average: " sprintf("%.1f", avg) "s"
  print "  Min: " sprintf("%.1f", min) "s"
  print "  Max: " sprintf("%.1f", max) "s"
  print "  Std Dev: " sprintf("%.1f", stddev) "s"
  print "  Variance: " sprintf("%.1f%%", (stddev/avg)*100)
}' /tmp/times.txt

echo ""
echo "Size vs Time Correlation:"
paste /tmp/sizes.txt /tmp/times.txt | awk '{
  size=$1/1048576; time=$2;
  sum_size+=size; sum_time+=time;
  sum_size_time+=size*time;
  sum_size_sq+=size*size;
  sum_time_sq+=time*time;
  n++
}
END {
  mean_size=sum_size/n;
  mean_time=sum_time/n;
  
  # Correlation coefficient
  numerator = sum_size_time - n*mean_size*mean_time;
  denom = sqrt((sum_size_sq - n*mean_size*mean_size) * (sum_time_sq - n*mean_time*mean_time));
  correlation = numerator/denom;
  
  print "  Correlation (size vs time): " sprintf("%.3f", correlation)
  if(correlation > 0.7) print "  Strong positive correlation - larger files take longer"
  else if(correlation > 0.3) print "  Moderate correlation"
  else print "  Weak correlation - time varies independently"
}'

echo ""
echo "=== Quality Margin Analysis ==="
# How much are we exceeding the minimum threshold?
tail -1000 ~/.pm2/logs/gen-worker-1-error.log | \
  grep -oP "Gaussians: \K[0-9,]+" | \
  tr -d ',' | \
  tail -30 | \
  awk '{
    threshold=150000;
    margin=($1-threshold)/threshold*100;
    sum_margin+=margin;
    count++;
    
    if(margin < 0) under++
    else if(margin < 50) low_margin++
    else if(margin < 100) good_margin++
    else if(margin < 200) high_margin++
    else extreme_margin++
  }
  END {
    avg_margin=sum_margin/count;
    print "Average margin above threshold: " sprintf("%.1f%%", avg_margin)
    print ""
    print "Margin Distribution:"
    print "  Below threshold (<0%):     " under " samples"
    print "  Close to threshold (0-50%): " low_margin " samples"  
    print "  Comfortable (50-100%):     " good_margin " samples"
    print "  High margin (100-200%):    " high_margin " samples"
    print "  Excessive (>200%):         " extreme_margin " samples"
    print ""
    
    if(avg_margin > 200) {
      print "⚠️  INSIGHT: Average margin is " sprintf("%.0f%%", avg_margin) " above threshold"
      print "   This suggests we might be OVER-OPTIMIZING for quality."
      print "   We could potentially reduce enhancement for faster generation."
    } else if(avg_margin > 100) {
      print "✅ INSIGHT: Healthy " sprintf("%.0f%%", avg_margin) " margin provides safety buffer"
      print "   Current configuration is well-balanced."
    } else if(avg_margin > 50) {
      print "⚠️  INSIGHT: Margin is comfortable but not excessive"
      print "   Current configuration is near-optimal."
    } else {
      print "❌ WARNING: Average margin only " sprintf("%.0f%%", avg_margin) " above threshold"
      print "   Risk of occasional failures. Should increase enhancement."
    }
  }'

echo ""
echo "=== Speed vs Quality Tradeoff Analysis ==="

# Calculate efficiency metric (gaussians per second)
tail -1000 ~/.pm2/logs/gen-worker-1-error.log | \
  grep "Gaussians:" | tail -30 | \
  grep -oP "Gaussians: \K[0-9,]+" | tr -d ',' > /tmp/gaussian_counts.txt

# Get corresponding generation times from validation
awk '{print $3}' /tmp/times.txt | sed 's/s$//' > /tmp/gen_times.txt

paste /tmp/gaussian_counts.txt /tmp/gen_times.txt 2>/dev/null | \
  awk '{
    if(NF==2) {
      gaussians=$1; time=$2;
      efficiency=gaussians/time;
      sum_eff+=efficiency;
      count++;
      
      if(efficiency > 30000) very_fast++
      else if(efficiency > 20000) fast++
      else if(efficiency > 15000) normal++
      else slow++
    }
  }
  END {
    if(count>0) {
      avg_eff=sum_eff/count;
      print "Average efficiency: " sprintf("%.0f", avg_eff) " gaussians/second"
      print ""
      print "Efficiency Distribution:"
      print "  Very fast (>30K/s):  " very_fast " generations"
      print "  Fast (20-30K/s):     " fast " generations"
      print "  Normal (15-20K/s):   " normal " generations"
      print "  Slow (<15K/s):       " slow " generations"
    }
  }'

echo ""
echo "=== Retry Logic Cost-Benefit Analysis ==="
echo ""

retry_attempts=$(tail -1000 ~/.pm2/logs/gen-worker-1-error.log | grep "🔄 Retry result" | wc -l)
total_gens=$(tail -1000 ~/.pm2/logs/gen-worker-1-error.log | grep "✅ TRELLIS generation done" | wc -l)

if [ $total_gens -gt 0 ]; then
  retry_rate=$(echo "scale=1; $retry_attempts * 100 / $total_gens" | bc)
  echo "Retry trigger rate: $retry_rate% ($retry_attempts out of $total_gens)"
  echo ""
  
  if (( $(echo "$retry_rate < 5" | bc -l) )); then
    echo "💡 INSIGHT: Retry logic rarely needed (<5%)"
    echo "   Current baseline enhancement is so effective that retry is mostly unused."
    echo "   Options:"
    echo "   1. Keep retry as safety net (recommended)"
    echo "   2. Remove retry to simplify code (minor gain)"
  elif (( $(echo "$retry_rate < 20" | bc -l) )); then
    echo "✅ INSIGHT: Retry logic provides good safety net"
    echo "   Retry catches $retry_rate% of borderline cases."
  else
    echo "⚠️  WARNING: High retry rate (>20%)"
    echo "   Baseline enhancement might be too weak."
  fi
else
  echo "No generation data found in recent logs."
fi

echo ""
echo "=== Prompt Complexity vs Output Quality ==="
echo ""

echo "Analyzing prompt patterns from validation..."
# Simple prompts (2-3 words)
simple_prompts="simple wooden cube|smooth metal sphere|clear glass bottle|simple silver ring"
# Complex prompts (4+ words)
complex_prompts="weathered brick wall section|chrome bicycle handlebar|ornate golden crown"

grep -E "$simple_prompts" validation_test_output.txt | grep "SUCCESS:" | \
  awk '{print $3}' | sed 's/MB$//' > /tmp/simple_sizes.txt

grep -E "$complex_prompts" validation_test_output.txt | grep "SUCCESS:" | \
  awk '{print $3}' | sed 's/MB$//' > /tmp/complex_sizes.txt

simple_avg=$(awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}' /tmp/simple_sizes.txt)
complex_avg=$(awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}' /tmp/complex_sizes.txt)

if [ -n "$simple_avg" ] && [ -n "$complex_avg" ]; then
  echo "Simple prompts average size: ${simple_avg}MB"
  echo "Complex prompts average size: ${complex_avg}MB"
  echo ""
  
  diff=$(echo "$complex_avg - $simple_avg" | bc)
  pct_diff=$(echo "scale=1; $diff * 100 / $simple_avg" | bc)
  
  if (( $(echo "$pct_diff > 20" | bc -l) )); then
    echo "💡 INSIGHT: Complex prompts generate ${pct_diff}% larger outputs"
    echo "   TRELLIS benefits significantly from detailed prompts."
  else
    echo "✅ INSIGHT: Prompt complexity has minimal impact"
    echo "   Enhancement preprocessing normalizes outputs across prompt types."
  fi
fi

echo ""
echo "=========================================="
echo "OPTIMIZATION RECOMMENDATIONS"
echo "=========================================="
echo ""

# Based on all analysis above, provide recommendations
cat << 'RECOMMENDATIONS'
Based on validation data analysis:

1. CURRENT CONFIGURATION (3.5x sharpness, 1.8x contrast):
   ✅ Excellent performance (100% success rate)
   ✅ Healthy safety margin (~216% above threshold)
   ⚠️  Might be slightly over-optimized for quality

2. OPTIMIZATION OPPORTUNITIES:

   Option A: MAINTAIN CURRENT (Recommended for mainnet start)
   - Keep 3.5x/1.8x for maximum reliability
   - Accept ~24s average generation time
   - Zero risk of sparse generations
   - Best choice for stable mainnet operation

   Option B: OPTIMIZE FOR SPEED (After mainnet is stable)
   - Reduce to 3.0x sharpness, 1.6x contrast
   - Potentially faster generation (~20s)
   - Still safe (should maintain >100% margin)
   - Test on testnet first before mainnet

   Option C: OPTIMIZE FOR EXTREME QUALITY (Not recommended)
   - Increase to 4.0x/2.0x baseline
   - Larger file sizes (>40MB average)
   - Slower generation (~30s+)
   - Diminishing returns - already at top tier

3. RETRY LOGIC:
   💡 Keep as safety net but it's rarely needed
   - Only 0% trigger rate in validation
   - Provides insurance against edge cases
   - Minimal overhead when not triggered

4. MONITORING PRIORITIES:
   📊 Track gaussian count distribution
   📊 Monitor actual validator scores (not just success rate)
   📊 Compare file sizes to top miners
   📊 Watch for any sparse generations (should be <5%)

RECOMMENDATIONS

