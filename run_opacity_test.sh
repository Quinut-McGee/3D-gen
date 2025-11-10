#!/bin/bash
# Phase 3A: Run 20-30 test generations to analyze opacity patterns

echo "=== PHASE 3A: OPACITY DIAGNOSTIC TEST ==="
echo "Running 30 test generations to analyze opacity corruption patterns..."
echo ""

# Test prompts (mix of simple and complex)
PROMPTS=(
    "red sports car"
    "blue wooden chair"
    "ornate golden chandelier with crystals"
    "vintage leather armchair with brass studs"
    "green apple"
    "silver fork"
    "mechanical steampunk pocket watch"
    "intricate Celtic knot sculpture"
    "yellow banana"
    "diamond engagement ring"
    "detailed marble statue of a lion"
    "hammer with wooden handle"
    "screwdriver with rubber grip"
    "pearl necklace"
    "adjustable wrench"
)

# Cycle through prompts to generate 30 total tests
COUNT=0
for i in {1..30}; do
    # Select prompt (cycle through array)
    PROMPT_INDEX=$((($i - 1) % ${#PROMPTS[@]}))
    PROMPT="${PROMPTS[$PROMPT_INDEX]}"

    echo "[$i/30] Testing: '$PROMPT'"

    # Call generation service
    curl -s -X POST http://localhost:10010/generate/ \
        -F "prompt=$PROMPT" \
        -o /tmp/opacity_test_$i.ply \
        --max-time 60

    if [ $? -eq 0 ]; then
        SIZE=$(stat -f%z /tmp/opacity_test_$i.ply 2>/dev/null || stat -c%s /tmp/opacity_test_$i.ply 2>/dev/null)
        echo "   ✅ Generated (${SIZE} bytes)"
        ((COUNT++))
    else
        echo "   ❌ Failed"
    fi

    # Small delay between requests
    sleep 2
    echo ""
done

echo ""
echo "=== TEST COMPLETE ==="
echo "Successfully generated: $COUNT/30"
echo ""
echo "📊 Now analyzing opacity patterns..."
echo "Check TRELLIS logs for diagnostic output:"
echo "   tail -200 /home/kobe/.pm2/logs/trellis-microservice-out.log | grep -A 4 '🔬'"
