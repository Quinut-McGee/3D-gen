#!/bin/bash
# Monitor mainnet mining activity in real-time

echo "======================================================================="
echo "🔍 MAINNET MINING MONITOR (UID 226)"
echo "======================================================================="
echo ""
echo "Watching for:"
echo "  - Task pulls from validators"
echo "  - Generation completions"
echo "  - Validator scores"
echo "  - Emission/incentive updates"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo "======================================================================="
echo ""

# Follow miner logs and filter for important events
tail -f ~/.pm2/logs/miner-sn17-mainnet-out.log | grep --line-buffered -E "Task pulled|Generated|Score|Emission|Incentive|UID:226|Feedback|Accepted|Rejected" --color=always
