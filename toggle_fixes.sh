#!/bin/bash
# Toggle fixes on/off for baseline comparison

MODE=$1

if [ "$MODE" == "disable" ]; then
    echo "🔧 Disabling all fixes for baseline test..."

    # Comment out opacity normalization in serve_trellis.py
    sed -i '/# 🔧 PHASE 3 FIX: Pre-save opacity normalization/,/# Continue anyway - downstream ply_fixer.py/s/^/# DISABLED_FOR_BASELINE: /' \
        /home/kobe/404-gen/v1/3D-gen/generation/serve_trellis.py

    # Restart TRELLIS
    pm2 restart trellis-microservice

    echo "✅ Fixes disabled (baseline mode)"
    echo "   - Opacity normalization: DISABLED"
    echo "   - Expecting 50-90% corruption rate"

elif [ "$MODE" == "enable" ]; then
    echo "🔧 Re-enabling all fixes..."

    # Uncomment opacity normalization
    sed -i 's/^# DISABLED_FOR_BASELINE: //' \
        /home/kobe/404-gen/v1/3D-gen/generation/serve_trellis.py

    # Restart TRELLIS
    pm2 restart trellis-microservice

    echo "✅ Fixes enabled (production mode)"
    echo "   - Opacity normalization: ENABLED"
    echo "   - Expecting 0% corruption rate"

else
    echo "Usage: $0 [enable|disable]"
    exit 1
fi

sleep 5  # Wait for TRELLIS to restart
