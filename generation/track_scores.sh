#!/bin/bash
# Real-time score tracking with running statistics

GOOD_SCORES=0
BAD_SCORES=0
START_TIME=$(date +%s)

echo "📊 SCORE TRACKER STARTED: $(date)"
echo "Tracking validator scores for 1 hour..."
echo "=========================================="
echo ""

tail -f ~/.pm2/logs/miner-sn17-mainnet-out.log | grep --line-buffered "Feedback from" | while read line; do
    # Extract score
    SCORE=$(echo "$line" | grep -oP "Score=\K[0-9.]+" | head -1)
    VALIDATOR=$(echo "$line" | grep -oP "Feedback from \[\K[0-9]+")
    
    if [ ! -z "$SCORE" ]; then
        # Check if score is 0.0 or > 0
        if [ "$SCORE" == "0.0" ]; then
            BAD_SCORES=$((BAD_SCORES + 1))
            echo "❌ [$(date +%H:%M:%S)] Validator $VALIDATOR → Score=0.0 | Running: $GOOD_SCORES good / $BAD_SCORES bad ($(awk "BEGIN {printf \"%.1f\", $GOOD_SCORES * 100 / ($GOOD_SCORES + $BAD_SCORES)}")%)"
        else
            GOOD_SCORES=$((GOOD_SCORES + 1))
            echo "✅ [$(date +%H:%M:%S)] Validator $VALIDATOR → Score=$SCORE | Running: $GOOD_SCORES good / $BAD_SCORES bad ($(awk "BEGIN {printf \"%.1f\", $GOOD_SCORES * 100 / ($GOOD_SCORES + $BAD_SCORES)}")%)"
        fi
        
        # Check if 1 hour has elapsed
        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - START_TIME))
        if [ $ELAPSED -ge 3600 ]; then
            echo ""
            echo "=========================================="
            echo "⏰ SCORE TRACKING COMPLETE: $(date)"
            echo "Duration: 1 hour"
            echo "Good scores: $GOOD_SCORES"
            echo "Bad scores (0.0): $BAD_SCORES"
            echo "Success rate: $(awk "BEGIN {printf \"%.1f\", $GOOD_SCORES * 100 / ($GOOD_SCORES + $BAD_SCORES)}")%"
            echo "=========================================="
            break
        fi
    fi
done
