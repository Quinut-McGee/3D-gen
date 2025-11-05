#!/bin/bash
# Phase 1 Validation Test: Opacity Fix + Memory Management + TRELLIS Quality
#
# This test validates:
# 1. Opacity variation is preserved (opacity_std should be 3-9, not 0-2)
# 2. Memory stays under 14GB (no degradation)
# 3. Gaussian counts are improved (>350K average)
#
# Run time: ~10 minutes (20 gens × 30s each)

set -e

echo "=========================================="
echo "Phase 1 Validation Test"
echo "Testing: Opacity fix + Memory + TRELLIS"
echo "=========================================="
echo ""

# 1. Restart services with new PM2 config
echo "Step 1: Restarting services with new configuration..."
pm2 restart trellis-microservice
sleep 5
pm2 restart gen-worker-1
sleep 10

echo "✅ Services restarted"
echo ""

# 2. Clear old logs
echo "Step 2: Clearing old logs..."
> /home/kobe/.pm2/logs/gen-worker-1-error.log
> /home/kobe/.pm2/logs/gen-worker-1-out.log

echo "✅ Logs cleared"
echo ""

# 3. Run 20 test generations
echo "Step 3: Running 20 test generations..."
echo "Test prompts:"

test_prompts=(
    "a vintage wooden chair"
    "a red sports car"
    "a glass wine bottle"
    "a medieval sword"
    "a ceramic coffee mug"
    "a leather backpack"
    "a golden crown"
    "a steel hammer"
    "a crystal ball"
    "a silver ring"
    "a bronze statue"
    "a plastic toy robot"
    "a bamboo flute"
    "a iron anvil"
    "a marble sculpture"
    "a copper kettle"
    "a wooden barrel"
    "a stone column"
    "a metal gear"
    "a fabric hat"
)

for i in "${!test_prompts[@]}"; do
    prompt="${test_prompts[$i]}"
    echo "  [$((i+1))/20] Testing: '$prompt'"

    # Call generation service (with trailing slash to avoid redirect)
    curl -s -L -X POST http://localhost:10010/generate/ \
        -F "prompt=$prompt" \
        -o /tmp/test_gen_$i.ply \
        --max-time 45

    # Brief pause between generations
    sleep 2
done

echo ""
echo "✅ 20 generations completed"
echo ""

# 4. Analyze results
echo "Step 4: Analyzing results..."
echo ""

# Check opacity variation preservation
echo "=== OPACITY ANALYSIS ==="
echo "Natural models should have opacity_std = 3.0-9.0"
echo "Flat models (BAD) have opacity_std = 0.0-2.0"
echo ""

opacity_fixes=$(grep "OPACITY FIXED" /home/kobe/.pm2/logs/gen-worker-1-error.log | wc -l)
echo "Opacity fixes triggered: $opacity_fixes"

if [ $opacity_fixes -gt 0 ]; then
    echo ""
    echo "Checking variation preservation:"
    grep "Variation PRESERVED" /home/kobe/.pm2/logs/gen-worker-1-error.log | tail -5
fi

echo ""
echo "=== MEMORY ANALYSIS ==="
memory_warnings=$(grep "HIGH MEMORY" /home/kobe/.pm2/logs/gen-worker-1-error.log | wc -l)
memory_critical=$(grep "CRITICAL MEMORY" /home/kobe/.pm2/logs/gen-worker-1-error.log | wc -l)
memory_skipped=$(grep "SKIPPING this generation" /home/kobe/.pm2/logs/gen-worker-1-error.log | wc -l)

echo "Memory warnings (>12GB): $memory_warnings"
echo "Memory critical (>14GB): $memory_critical"
echo "Generations skipped due to memory: $memory_skipped"

if [ $memory_critical -gt 0 ]; then
    echo "⚠️  WARNING: Memory exceeded 14GB - may need more aggressive cleanup"
fi

echo ""
echo "=== GAUSSIAN QUALITY ANALYSIS ==="
echo "Target: Average >350K gaussians"
echo ""

# Extract gaussian counts and calculate stats
grep "Gaussians:" /home/kobe/.pm2/logs/gen-worker-1-error.log | tail -20 | \
    awk -F'Gaussians: ' '{print $2}' | \
    awk -F',' '{
        gsub(/,/, "", $1)
        count = $1 + 0
        if (count > 0) {
            sum += count
            n++
            if (count > 400000) high++
            else if (count >= 150000) med++
            else low++

            if (n == 1 || count < min) min = count
            if (n == 1 || count > max) max = count
        }
    } END {
        if (n > 0) {
            avg = int(sum / n)
            print "Generations: " n
            print "Average: " avg " gaussians"
            print "Range: " min " - " max
            print ""
            print "Distribution:"
            print "  High (>400K): " high " (" int(high/n*100) "%)"
            print "  Medium (150-400K): " med " (" int(med/n*100) "%)"
            print "  Low (<150K): " low " (" int(low/n*100) "%)"
            print ""

            if (avg > 350000) {
                print "✅ PASS: Average gaussian count meets target"
            } else {
                print "⚠️  WARNING: Average below target (350K)"
            }
        }
    }'

echo ""
echo "=== GENERATION TIME ANALYSIS ==="
echo "Target: <35 seconds per generation"
echo ""

# Check generation times
grep "Generation completed in" /home/kobe/.pm2/logs/trellis-microservice-error.log | tail -20 | \
    awk -F'in ' '{print $2}' | \
    awk -F's' '{
        time = $1 + 0
        if (time > 0) {
            sum += time
            n++
            if (time > max) max = time
            if (n == 1 || time < min) min = time
        }
    } END {
        if (n > 0) {
            avg = sum / n
            printf "Average: %.2fs\n", avg
            printf "Range: %.2fs - %.2fs\n", min, max

            if (avg < 35) {
                print "✅ PASS: Generation time within budget"
            } else {
                print "⚠️  WARNING: Generation time exceeds 35s budget"
            }
        }
    }'

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Next Steps:"
echo "1. Review opacity variation (should be preserved)"
echo "2. Confirm memory stayed under 14GB"
echo "3. Check gaussian counts improved (>350K avg)"
echo "4. If all pass, proceed to Phase 2 (mainnet testing)"
echo ""
