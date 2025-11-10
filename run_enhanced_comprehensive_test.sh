#!/bin/bash
# Enhanced 130-Sample Comprehensive Test with Baseline Comparison
# Total: 120 samples (4 tiers) + 10 baseline samples
# Estimated time: 70-95 minutes

echo "=== ENHANCED 130-SAMPLE COMPREHENSIVE TEST ==="
echo "Testing across 5 tiers with intermediate checkpoints"
echo ""

# Create results directory
mkdir -p /tmp/tier_test_results
echo '{"metadata": {"test_start": "'$(date -Iseconds)'", "total_planned": 130}}' > /tmp/tier_test_results/test_metadata.json

# Tier 5: Baseline (NO FIXES - 10 samples to prove fixes work)
TIER5_BASELINE=(
    "red sports car"
    "vintage wooden rocking chair with cushions"
    "detailed victorian era steampunk mechanical pocket watch with brass gears"
    "highly detailed artisan crafted oak coffee table with intricate carved scrollwork and brass inlay accents throughout"
    "blue vase"
    "ornate gold chandelier with crystals"
    "intricately carved mahogany bookshelf filled with leather bound vintage books"
    "vintage industrial factory gear assembly with weathered cast iron components and exposed mechanical linkages fully detailed"
    "golden ring"
    "rustic farmhouse dining table with details"
)

# Tier 1: Simple Prompts (1-3 words, 30 samples)
TIER1_PROMPTS=(
    "red car"
    "wooden chair"
    "blue vase"
    "metal sword"
    "glass bottle"
    "leather boot"
    "stone statue"
    "golden ring"
    "silver watch"
    "ceramic mug"
    "iron hammer"
    "bronze bell"
    "crystal orb"
    "wooden box"
    "steel knife"
    "copper pot"
    "marble bust"
    "glass lamp"
    "clay bowl"
    "gold coin"
    "silver fork"
    "jade pendant"
    "brass compass"
    "oak barrel"
    "steel sword"
    "copper wire"
    "stone block"
    "iron chain"
    "bronze statue"
    "pearl necklace"
)

# Tier 2: Moderate Prompts (4-8 words, 30 samples)
TIER2_PROMPTS=(
    "vintage wooden rocking chair with cushions"
    "ornate gold chandelier with crystals"
    "rustic farmhouse dining table with details"
    "sleek modern office desk lamp"
    "antique bronze telescope on tripod"
    "hand-painted ceramic vase with flowers"
    "weathered leather messenger bag with buckles"
    "polished mahogany grandfather clock face"
    "decorative wrought iron garden bench seat"
    "elegant crystal wine decanter with stopper"
    "vintage brass nautical compass in box"
    "carved wooden chess set with pieces"
    "stained glass window panel with colors"
    "copper tea kettle with wooden handle"
    "stone garden fountain with basin"
    "wrought iron candle holder with arms"
    "wooden jewelry box with velvet interior"
    "brass telescope with leather grip"
    "ceramic flower pot with drainage hole"
    "silver tea service on tray"
    "oak bookshelf with glass doors"
    "bronze door knocker with ring"
    "marble column with capital top"
    "leather bound journal with clasp"
    "brass pocket watch with chain"
    "wooden rocking horse with saddle"
    "iron gate with decorative scrollwork"
    "crystal chandelier with hanging prisms"
    "copper weathervane with rooster design"
    "stone bench with carved armrests"
)

# Tier 3: Complex Prompts (9-12 words, 30 samples)
TIER3_PROMPTS=(
    "detailed victorian era steampunk mechanical pocket watch with brass gears"
    "intricately carved mahogany bookshelf filled with leather bound vintage books"
    "ornate french provincial writing desk with gold leaf trim and drawers"
    "rustic farmhouse kitchen island with butcher block top and storage"
    "vintage brass nautical telescope mounted on wooden tripod stand base"
    "elegant art deco chrome table lamp with frosted glass shade"
    "weathered antique leather armchair with brass studded trim details"
    "hand-painted porcelain tea set with delicate floral pattern design"
    "baroque style gilded mirror frame with elaborate scrollwork and details"
    "medieval knight armor suit standing on wooden display pedestal"
    "ornate victorian era music box with rotating ballerina figure inside"
    "vintage industrial factory gear assembly with cast iron and brass"
    "antique apothecary cabinet with labeled drawers and brass hardware"
    "rustic wooden wagon wheel with iron rim and wooden spokes"
    "decorative stained glass lamp shade with colorful floral motifs"
    "carved wooden chess board with detailed figurine pieces on squares"
    "vintage brass microscope on mahogany base with adjustment knobs"
    "ornate cast iron fireplace screen with decorative flourishes and detail"
    "weathered wooden ship wheel with brass center hub and handles"
    "elegant crystal perfume bottle with ornate silver cap and atomizer"
    "antique brass sextant navigation instrument with telescopic sight attached"
    "carved jade dragon statue on wooden display stand with details"
    "vintage copper still apparatus with coiled tubing and glass vessels"
    "ornate bronze candelabra with multiple arms and decorative base plate"
    "rustic leather saddle with brass buckles and tooled patterns"
    "antique wooden spinning wheel with large wheel and spindle mechanism"
    "decorative porcelain vase with hand-painted oriental landscape scenes"
    "vintage brass diving helmet with glass portholes and valve hardware"
    "carved wooden totem pole with multiple stacked animal figures"
    "ornate silver tea service set with engraved patterns and handles"
)

# Tier 4: Very Complex Prompts (13-16 words, 30 samples)
TIER4_PROMPTS=(
    "highly detailed artisan crafted oak coffee table with intricate carved scrollwork and brass inlay accents throughout"
    "vintage industrial factory gear assembly with weathered cast iron components and exposed mechanical linkages fully detailed"
    "ornate renaissance period wooden writing desk featuring multiple drawers hand-carved details and antique brass hardware accents"
    "weathered ancient stone fountain sculpture with multiple tiers moss covered surfaces and decorative carved mythological figures"
    "elaborate victorian era brass microscope on mahogany stand with precision adjustment knobs and original glass optics"
    "rustic reclaimed barn wood farmhouse dining table with natural weathering thick planks and wrought iron support brackets"
    "antique grandfather clock in mahogany case with brass pendulum roman numeral face and ornate carved crown details"
    "medieval castle stone archway entrance with weathered blocks iron gate hinges and carved heraldic emblems above"
    "vintage apothecary cabinet with dozens of small labeled drawers brass pulls and distressed painted wood finish"
    "steampunk mechanical automaton figure with visible brass gears copper pipes and articulated limb joints exposed"
    "ornate baroque style gilded picture frame with elaborate three dimensional scrollwork acanthus leaves and cherub details"
    "weathered nautical ship anchor made of heavy forged iron with thick rope chain and barnacle encrusted surface texture"
    "intricately detailed mechanical clockwork mechanism with exposed brass gears springs jeweled bearings and pendulum assembly"
    "rustic medieval blacksmith anvil on wooden stump base with hammer marks wear patterns and tool marks on surface"
    "elegant art nouveau style stained glass window panel featuring flowing organic floral patterns in vibrant jewel tone colors"
    "antique Victorian era phonograph with large brass horn curved wooden base and hand crank mechanism fully functional"
    "ornate gothic cathedral rose window with intricate stone tracery colorful stained glass panels and religious iconography scenes"
    "weathered vintage leather aviator jacket with fur collar brass zipper multiple pockets and distressed brown patina"
    "detailed steampunk airship propeller assembly with massive brass blades riveted metal framework and gear drive mechanism"
    "carved wooden carousel horse with elaborate saddle bridle decorative brass pole mount and hand-painted colorful details"
    "antique brass ship sextant navigation instrument in mahogany case with telescopic sight mirrors and degree markings"
    "ornate renaissance era globe on wooden stand with hand-drawn continents sea monsters brass meridian ring details"
    "vintage industrial cast iron gear mechanism with multiple interlocking cogs chain drives and oil-stained metal surfaces"
    "weathered stone gargoyle statue with wings open mouth detailed scales and moss-covered surface mounted on pedestal"
    "elaborate Victorian era music box with rotating cylinder pin mechanism inlaid wooden case and hand-painted porcelain figurines"
    "antique wooden ship's wheel with eight spokes brass center hub rope wrapping and weathered varnish finish"
    "ornate bronze fountain sculpture featuring multiple mythological figures water spouts tiered basins and verdigris patina surface"
    "vintage brass diving suit helmet with round glass portholes air valves copper rivets and aged metal patina"
    "detailed steampunk laboratory apparatus with glass tubes copper coils pressure gauges brass fittings and bubbling liquid effects"
    "carved wooden totem pole with stacked animal spirits eagle bear salmon detailed paint colors and weathered wood texture"
)

# Test execution
TOTAL=130
COUNT=0
SUCCESS=0
BASELINE_START=0
TIER1_START=0
TIER2_START=0
TIER3_START=0
TIER4_START=0

# Checkpoint function
run_checkpoint_analysis() {
    local TIER_NAME="$1"
    local TIER_NUM="$2"

    echo ""
    echo "📊 CHECKPOINT: $TIER_NAME Complete"
    echo "=========================================="

    # Quick analysis
    RECENT_GENS=$(tail -200 /home/kobe/.pm2/logs/trellis-microservice-out.log | grep -c "RAW TRELLIS OPACITY")
    RECENT_NORM=$(tail -200 /home/kobe/.pm2/logs/trellis-microservice-out.log | grep -c "🔧")
    RECENT_INF=$(tail -200 /home/kobe/.pm2/logs/trellis-microservice-out.log | grep "SAVED PLY" -A 1 | grep -c "inf")

    echo "  Recent generations: $RECENT_GENS"
    echo "  Normalizations applied: $RECENT_NORM"
    echo "  Inf/nan detected: $RECENT_INF"

    if [ $RECENT_INF -gt 0 ]; then
        echo "  ⚠️  WARNING: Corruption detected! Investigate immediately."
    else
        echo "  ✅ No corruption detected"
    fi

    # Save checkpoint
    echo "{\"tier\": \"$TIER_NAME\", \"completed\": $COUNT, \"corruption\": $RECENT_INF}" > /tmp/tier_test_results/checkpoint_tier${TIER_NUM}.json

    echo "=========================================="
    echo ""
}

# Function to test a prompt
test_prompt() {
    local PROMPT="$1"
    local TIER="$2"
    local INDEX="$3"

    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] Tier $TIER: '$PROMPT'"

    START_TIME=$(date +%s.%N)

    # Call generation service
    RESPONSE=$(curl -s -X POST http://localhost:10010/generate/ \
        -F "prompt=$PROMPT" \
        -o /tmp/tier_test_results/tier${TIER}_${INDEX}.ply \
        -w "\n%{http_code}" \
        --max-time 60 \
        2>&1)

    END_TIME=$(date +%s.%N)
    GEN_TIME=$(echo "$END_TIME - $START_TIME" | bc)

    if [ $? -eq 0 ]; then
        SIZE=$(stat -c%s /tmp/tier_test_results/tier${TIER}_${INDEX}.ply 2>/dev/null || echo "0")
        if [ $SIZE -gt 1000 ]; then
            echo "   ✅ Generated (${SIZE} bytes, ${GEN_TIME}s)"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "   ⚠️  Filtered by quality gate (${SIZE} bytes, ${GEN_TIME}s)"
        fi
    else
        echo "   ❌ Failed"
    fi

    # Small delay
    sleep 2
    echo ""
}

# ===== TIER 5: BASELINE (NO FIXES) =====
echo "=========================================="
echo "TIER 5: BASELINE (NO FIXES - 10 samples)"
echo "=========================================="
echo "Temporarily disabling opacity normalization fix..."
bash toggle_fixes.sh disable
sleep 5

echo ""
echo "Running baseline samples (expecting 50-90% corruption)..."
echo ""

BASELINE_START=$COUNT
for i in "${!TIER5_BASELINE[@]}"; do
    test_prompt "${TIER5_BASELINE[$i]}" "5_baseline" "$i"
done

# Re-enable fixes
echo ""
echo "Re-enabling fixes for production tiers..."
bash toggle_fixes.sh enable
sleep 5

run_checkpoint_analysis "Tier 5: Baseline (NO FIXES)" "5"

# ===== TIER 1: SIMPLE =====
echo ""
echo "=========================================="
echo "TIER 1: SIMPLE PROMPTS (30 samples)"
echo "=========================================="
TIER1_START=$COUNT
for i in "${!TIER1_PROMPTS[@]}"; do
    test_prompt "${TIER1_PROMPTS[$i]}" "1" "$i"
done
run_checkpoint_analysis "Tier 1: Simple" "1"

# ===== TIER 2: MODERATE =====
echo ""
echo "=========================================="
echo "TIER 2: MODERATE PROMPTS (30 samples)"
echo "=========================================="
TIER2_START=$COUNT
for i in "${!TIER2_PROMPTS[@]}"; do
    test_prompt "${TIER2_PROMPTS[$i]}" "2" "$i"
done
run_checkpoint_analysis "Tier 2: Moderate" "2"

# ===== TIER 3: COMPLEX =====
echo ""
echo "=========================================="
echo "TIER 3: COMPLEX PROMPTS (30 samples)"
echo "=========================================="
TIER3_START=$COUNT
for i in "${!TIER3_PROMPTS[@]}"; do
    test_prompt "${TIER3_PROMPTS[$i]}" "3" "$i"
done
run_checkpoint_analysis "Tier 3: Complex" "3"

# ===== TIER 4: VERY COMPLEX =====
echo ""
echo "=========================================="
echo "TIER 4: VERY COMPLEX PROMPTS (30 samples)"
echo "=========================================="
TIER4_START=$COUNT
for i in "${!TIER4_PROMPTS[@]}"; do
    test_prompt "${TIER4_PROMPTS[$i]}" "4" "$i"
done
run_checkpoint_analysis "Tier 4: Very Complex" "4"

# ===== FINAL SUMMARY =====
echo ""
echo "=========================================="
echo "=== TEST COMPLETE ==="
echo "=========================================="
echo "Total: $COUNT/$TOTAL"
echo "Generated: $SUCCESS"
echo "Baseline (no fixes): 10"
echo "Tier 1 (simple): 30"
echo "Tier 2 (moderate): 30"
echo "Tier 3 (complex): 30"
echo "Tier 4 (very complex): 30"
echo ""
echo "📊 Run enhanced analysis:"
echo "   bash analyze_enhanced_tier_test.sh"
echo ""
echo "Test completed at: $(date)"
