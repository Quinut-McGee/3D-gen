#!/bin/bash
# Phase 3A: Analyze opacity corruption patterns from test generations

echo "=== PHASE 3A: OPACITY CORRUPTION ANALYSIS ==="
echo ""

# Count total generations with diagnostic data
TOTAL=$(tail -1000 /home/kobe/.pm2/logs/trellis-microservice-out.log | \
  grep -c "RAW TRELLIS OPACITY")

echo "1. Total generations analyzed: $TOTAL"
echo ""

if [ $TOTAL -eq 0 ]; then
    echo "❌ No diagnostic data found!"
    echo "   Make sure TRELLIS microservice is running and test generations completed."
    echo "   Check logs: tail -100 /home/kobe/.pm2/logs/trellis-microservice-out.log"
    exit 1
fi

# Count corrupted at source (mean < 4.0)
echo "2. Analyzing raw TRELLIS opacity values:"
tail -1000 /home/kobe/.pm2/logs/trellis-microservice-out.log | \
  grep "RAW TRELLIS OPACITY" -A 1 | \
  grep "mean:" | \
  awk 'BEGIN {total=0; corrupt=0; sum=0; neg_count=0; extreme_count=0}
       {total++; mean=$3+0; sum+=mean;
        if (mean < 4.0) {
            corrupt++;
            if (mean < 0) neg_count++;
            if (mean < -10 || mean > 20) extreme_count++;
            print "   🔴 CORRUPTED: mean=" sprintf("%.4f", mean)
        }}
       END {if (total>0) {
         print "";
         print "   📊 Summary:";
         print "      Total samples: " total;
         print "      Corrupted at source (mean < 4.0): " corrupt " (" int(corrupt*100/total) "%)";
         print "      Negative mean: " neg_count " (" int(neg_count*100/total) "%)";
         print "      Extreme values (<-10 or >20): " extreme_count " (" int(extreme_count*100/total) "%)";
         print "      Average mean: " sprintf("%.4f", sum/total);
       }}'

echo ""

# Check for corruption during save
echo "3. Checking corruption during save_ply():"
SAVE_CORRUPT=$(tail -1000 /home/kobe/.pm2/logs/trellis-microservice-out.log | \
  grep -c "CORRUPTION DETECTED")
echo "   Corrupted during save: $SAVE_CORRUPT / $TOTAL"

if [ $SAVE_CORRUPT -gt 0 ]; then
    echo ""
    echo "   Examples of save corruption:"
    tail -1000 /home/kobe/.pm2/logs/trellis-microservice-out.log | \
      grep "CORRUPTION DETECTED" -B 10 | \
      grep -E "RAW TRELLIS|corruption_delta" | \
      head -10
fi

echo ""

# Show distribution of opacity values
echo "4. Opacity distribution (last 30 samples):"
tail -1000 /home/kobe/.pm2/logs/trellis-microservice-out.log | \
  grep "RAW TRELLIS OPACITY" -A 1 | \
  grep "mean:" | \
  tail -30 | \
  awk '{mean=$3+0;
        if (mean < 0) print "   🔴 " mean " (NEGATIVE)";
        else if (mean < 4.0) print "   🟡 " mean " (LOW)";
        else if (mean > 10.0) print "   🟠 " mean " (HIGH)";
        else print "   🟢 " mean " (HEALTHY)";
       }'

echo ""

# Check for inf/nan values
echo "5. Checking for inf/nan values:"
INF_COUNT=$(tail -1000 /home/kobe/.pm2/logs/trellis-microservice-out.log | \
  grep "num_inf:" | \
  awk '{inf=$2+0; nan=$4+0; if (inf>0 || nan>0) print}' | \
  wc -l)
echo "   Generations with inf/nan: $INF_COUNT / $TOTAL"

echo ""
echo "=== DIAGNOSIS ==="

# Determine which scenario we're in
CORRUPT_COUNT=$(tail -1000 /home/kobe/.pm2/logs/trellis-microservice-out.log | \
  grep "RAW TRELLIS OPACITY" -A 1 | \
  grep "mean:" | \
  awk '{mean=$3+0; if (mean < 4.0) count++} END {print count+0}')

CORRUPT_RATE=$((CORRUPT_COUNT * 100 / TOTAL))

echo "Corruption rate at TRELLIS source: ${CORRUPT_RATE}%"
echo ""

if [ $CORRUPT_RATE -gt 20 ]; then
    echo "🎯 SCENARIO A: High Source Corruption Detected (>20%)"
    echo ""
    echo "   ✅ Implement Option 1b: Pre-save normalization in serve_trellis.py"
    echo ""
    echo "   This will fix opacity corruption BEFORE save_ply(), preventing:"
    echo "   - Negative opacity values"
    echo "   - Extreme outliers"
    echo "   - inf/nan values"
    echo ""
    echo "   Expected impact:"
    echo "   - Corruption rate: ${CORRUPT_RATE}% → <10%"
    echo "   - Success rate: +5-7% (reaching 83-85% target)"
elif [ $SAVE_CORRUPT -gt 0 ]; then
    echo "🎯 SCENARIO B: Corruption During save_ply() Detected"
    echo ""
    echo "   ⚠️  Investigate GaussianSplattingModel.py save method"
    echo ""
    echo "   Raw TRELLIS opacities are healthy, but corruption happens during save."
    echo "   This suggests a bug in the save_ply() serialization process."
elif [ $CORRUPT_RATE -lt 5 ]; then
    echo "🎯 SCENARIO C: No Significant TRELLIS-Level Corruption (<5%)"
    echo ""
    echo "   ✓ Current ply_fixer.py is handling corruption correctly"
    echo ""
    echo "   Corruption happens downstream (during load_ply in trellis_integration.py)."
    echo "   Your existing fix is already optimal - no changes needed at TRELLIS level."
else
    echo "🎯 MODERATE Corruption Detected (${CORRUPT_RATE}%)"
    echo ""
    echo "   Consider implementing Option 1b (pre-save normalization)"
    echo "   This will provide additional safety and reduce downstream processing."
fi

echo ""
echo "=== NEXT STEPS ==="
echo ""
echo "1. Review the opacity distribution above"
echo "2. Based on the diagnosis, choose the appropriate fix:"
echo "   - Option 1a: Increase TRELLIS sampling steps (80→100, 60→80)"
echo "   - Option 1b: Add pre-save normalization in serve_trellis.py"
echo "   - None: Current ply_fixer.py is sufficient"
echo ""
echo "3. View detailed logs:"
echo "   tail -200 /home/kobe/.pm2/logs/trellis-microservice-out.log | grep -A 4 '🔬'"
echo ""
