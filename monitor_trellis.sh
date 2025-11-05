#!/bin/bash
# Monitor TRELLIS submissions and track success metrics

LOG_FILE="/home/kobe/.pm2/logs/gen-worker-1-error.log"
MINER_LOG="/home/kobe/.pm2/logs/miner-sn17-mainnet-out.log"
REPORT_FILE="/tmp/trellis_monitoring_report.txt"

echo "=== TRELLIS MONITORING STARTED: $(date) ===" > "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "Tracking next 20 submissions..." >> "$REPORT_FILE"
echo "Monitoring for:" >> "$REPORT_FILE"
echo "  1. Gaussian counts (min/max/avg)" >> "$REPORT_FILE"
echo "  2. Validator feedback scores" >> "$REPORT_FILE"
echo "  3. Opacity corruption detection" >> "$REPORT_FILE"
echo "  4. Prompt complexity patterns" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "============================================================" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Extract recent TRELLIS generations
tail -n 2000 "$LOG_FILE" | grep -E "Detected TEXT-TO-3D task:|Gaussians:|opacity_std" | tail -60 >> "$REPORT_FILE"

echo "" >> "$REPORT_FILE"
echo "Recent validator feedback:" >> "$REPORT_FILE"
tail -n 100 "$MINER_LOG" | grep -E "Task queued|Score|Feedback" | tail -20 >> "$REPORT_FILE"

cat "$REPORT_FILE"
