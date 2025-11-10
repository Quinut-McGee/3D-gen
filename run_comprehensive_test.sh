#!/bin/bash
# 120-Sample Comprehensive Test: Validate Performance Across All Prompt Tiers
# Estimated time: 60-90 minutes

echo "=== 120-SAMPLE COMPREHENSIVE TIER TEST ==="
echo "Testing opacity normalization, quality gates, and performance across 4 complexity tiers"
echo ""

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
TOTAL=120
COUNT=0
SUCCESS=0
TIER1_COUNT=0
TIER2_COUNT=0
TIER3_COUNT=0
TIER4_COUNT=0

# Create results directory
mkdir -p /tmp/tier_test_results
echo "[]" > /tmp/tier_test_results/generations.json

# Function to test a prompt
test_prompt() {
    local PROMPT="$1"
    local TIER="$2"
    local INDEX="$3"

    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] Tier $TIER: '$PROMPT'"

    START_TIME=$(date +%s.%N)

    # Call generation service
    curl -s -X POST http://localhost:10010/generate/ \
        -F "prompt=$PROMPT" \
        -o /tmp/tier_test_results/tier${TIER}_${INDEX}.ply \
        --max-time 60 \
        2>&1 > /dev/null

    END_TIME=$(date +%s.%N)
    GEN_TIME=$(echo "$END_TIME - $START_TIME" | bc)

    if [ $? -eq 0 ]; then
        SIZE=$(stat -c%s /tmp/tier_test_results/tier${TIER}_${INDEX}.ply 2>/dev/null || echo "0")
        if [ $SIZE -gt 1000 ]; then
            echo "   ✅ Generated (${SIZE} bytes, ${GEN_TIME}s)"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "   ⚠️  Filtered by quality gate (${SIZE} bytes)"
        fi
    else
        echo "   ❌ Failed"
    fi

    # Update tier counter
    case $TIER in
        1) TIER1_COUNT=$((TIER1_COUNT + 1)) ;;
        2) TIER2_COUNT=$((TIER2_COUNT + 1)) ;;
        3) TIER3_COUNT=$((TIER3_COUNT + 1)) ;;
        4) TIER4_COUNT=$((TIER4_COUNT + 1)) ;;
    esac

    # Small delay
    sleep 2
    echo ""
}

# Run all tiers
echo "Starting Tier 1: Simple Prompts (1-3 words)"
echo "============================================"
for i in "${!TIER1_PROMPTS[@]}"; do
    test_prompt "${TIER1_PROMPTS[$i]}" "1" "$i"
done

echo ""
echo "Starting Tier 2: Moderate Prompts (4-8 words)"
echo "============================================"
for i in "${!TIER2_PROMPTS[@]}"; do
    test_prompt "${TIER2_PROMPTS[$i]}" "2" "$i"
done

echo ""
echo "Starting Tier 3: Complex Prompts (9-12 words)"
echo "============================================"
for i in "${!TIER3_PROMPTS[@]}"; do
    test_prompt "${TIER3_PROMPTS[$i]}" "3" "$i"
done

echo ""
echo "Starting Tier 4: Very Complex Prompts (13-16 words)"
echo "============================================"
for i in "${!TIER4_PROMPTS[@]}"; do
    test_prompt "${TIER4_PROMPTS[$i]}" "4" "$i"
done

echo ""
echo "=== TEST COMPLETE ==="
echo "Total: $COUNT/$TOTAL"
echo "Generated: $SUCCESS"
echo "Tier 1: $TIER1_COUNT"
echo "Tier 2: $TIER2_COUNT"
echo "Tier 3: $TIER3_COUNT"
echo "Tier 4: $TIER4_COUNT"
echo ""
echo "📊 Run analysis script to get detailed metrics:"
echo "   bash analyze_tier_test.sh"
