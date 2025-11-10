#!/bin/bash
# Quick performance check for mainnet mining

echo "======================================================================="
echo "📊 MAINNET PERFORMANCE SUMMARY"
echo "======================================================================="
echo ""

# Get last 100 log lines
LOG_LINES=$(tail -100 ~/.pm2/logs/miner-sn17-mainnet-out.log)

# Extract recent stats
RECENT_UID=$(echo "$LOG_LINES" | grep "UID:226" | tail -1)

if [ -n "$RECENT_UID" ]; then
    echo "📍 Latest Status:"
    echo "$RECENT_UID"
    echo ""
fi

# Count scores (if any)
TOTAL_SUBMISSIONS=$(echo "$LOG_LINES" | grep -c "Score:" || echo "0")
SUCCESSFUL=$(echo "$LOG_LINES" | grep "Score:" | grep -v "Score: 0\.0" -c || echo "0")

if [ "$TOTAL_SUBMISSIONS" -gt 0 ]; then
    SUCCESS_RATE=$(echo "scale=1; $SUCCESSFUL * 100 / $TOTAL_SUBMISSIONS" | bc)
    echo "📈 Submission Stats:"
    echo "  Total submissions: $TOTAL_SUBMISSIONS"
    echo "  Successful: $SUCCESSFUL"
    echo "  Success rate: ${SUCCESS_RATE}%"
    echo ""

    echo "📋 Recent scores:"
    echo "$LOG_LINES" | grep "Score:" | tail -10
else
    echo "⏳ Waiting for first task from validators..."
    echo ""
    echo "💡 This is normal for new miners. Validators will assign tasks shortly."
fi

echo ""
echo "======================================================================="
echo "🔧 Quick Commands:"
echo "======================================================================="
echo "  Monitor live:      ./monitor_mainnet.sh"
echo "  Check performance: ./check_performance.sh"
echo "  View all logs:     pm2 logs miner-sn17-mainnet"
echo "  Check services:    pm2 status"
echo "======================================================================="
