#!/bin/bash
# Periodic check-in every 10 minutes for 1 hour (6 check-ins total)

CHECKIN_INTERVAL=600  # 10 minutes
TOTAL_CHECKINS=6
START_TIME=$(date +%s)

for i in $(seq 1 $CHECKIN_INTERVAL); do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    REMAINING=$((3600 - ELAPSED))
    
    if [ $REMAINING -le 0 ]; then
        break
    fi
    
    echo "=========================================="
    echo "📋 CHECK-IN #$i of $TOTAL_CHECKINS"
    echo "Time: $(date)"
    echo "Elapsed: $(($ELAPSED / 60))m | Remaining: $(($REMAINING / 60))m"
    echo "=========================================="
    echo ""
    
    # Check service status
    echo "🔍 Service Status:"
    pm2 status | grep -E "gen-worker-1|miner-sn17-mainnet|trellis-microservice"
    echo ""
    
    # Check for crashed services
    CRASHED=$(pm2 status | grep -E "gen-worker-1|miner-sn17-mainnet|trellis-microservice" | grep -c "stopped\|errored")
    if [ $CRASHED -gt 0 ]; then
        echo "⚠️  ALERT: $CRASHED service(s) crashed! Restarting..."
        pm2 restart all
        echo "✅ Services restarted"
    else
        echo "✅ All services online"
    fi
    echo ""
    
    # Show current scores
    echo "📊 Current Score Stats:"
    tail -3 /tmp/score_tracking.log
    echo ""
    
    # Check for critical errors
    ERROR_COUNT=$(wc -l < /tmp/critical_errors.log)
    if [ $ERROR_COUNT -gt 10 ]; then
        echo "⚠️  WARNING: $ERROR_COUNT critical errors detected!"
        echo "Latest errors:"
        tail -5 /tmp/critical_errors.log
    else
        echo "✅ No critical errors"
    fi
    echo ""
    
    # Memory check
    WORKER_MEM=$(pm2 jlist | jq -r '.[] | select(.name=="gen-worker-1") | .monit.memory' | awk '{print int($1/1024/1024/1024)}')
    echo "💾 Worker Memory: ${WORKER_MEM}GB"
    if [ $WORKER_MEM -gt 40 ]; then
        echo "⚠️  WARNING: High memory usage! Consider restart if continues to grow"
    fi
    echo ""
    
    echo "⏳ Next check-in in 10 minutes..."
    echo ""
    
    sleep $CHECKIN_INTERVAL
done

echo "=========================================="
echo "✅ PERIODIC CHECK-INS COMPLETE"
echo "=========================================="
