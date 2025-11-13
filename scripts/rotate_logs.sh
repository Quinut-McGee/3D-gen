#!/bin/bash
# Rotate generation logs monthly and backup to archive
#
# Usage:
#   bash scripts/rotate_logs.sh
#
# Cron job (run monthly):
#   0 0 1 * * /home/kobe/404-gen/v1/3D-gen/scripts/rotate_logs.sh

DATA_DIR="/home/kobe/404-gen/v1/3D-gen/data"
ARCHIVE_DIR="$DATA_DIR/archive"

echo "========================================================================"
echo "MONTHLY LOG ROTATION"
echo "========================================================================"
echo ""

# Get current month
CURRENT_MONTH=$(date +"%Y-%m")

# Check if generation_history.jsonl exists
if [ ! -f "$DATA_DIR/generation_history.jsonl" ]; then
    echo "ℹ️  No generation_history.jsonl found - nothing to rotate"
    exit 0
fi

# Get first line to determine log month
FIRST_LINE=$(head -1 "$DATA_DIR/generation_history.jsonl")
if [ -z "$FIRST_LINE" ]; then
    echo "ℹ️  Empty generation_history.jsonl - nothing to rotate"
    exit 0
fi

# Extract timestamp from first entry
FIRST_TIMESTAMP=$(echo "$FIRST_LINE" | python3 -c "import json, sys; print(json.load(sys.stdin).get('metadata', {}).get('timestamp', ''))")
if [ -z "$FIRST_TIMESTAMP" ]; then
    echo "⚠️  Could not extract timestamp from first entry"
    exit 1
fi

# Get month from timestamp
FIRST_MONTH=$(date -d "$FIRST_TIMESTAMP" +"%Y-%m" 2>/dev/null)
if [ -z "$FIRST_MONTH" ]; then
    echo "⚠️  Could not parse timestamp: $FIRST_TIMESTAMP"
    exit 1
fi

echo "Current month: $CURRENT_MONTH"
echo "Log file starts: $FIRST_MONTH"
echo ""

# If log is from previous month, rotate it
if [ "$FIRST_MONTH" != "$CURRENT_MONTH" ]; then
    echo "🔄 Rotating logs from $FIRST_MONTH..."

    # Move to monthly archive
    mv "$DATA_DIR/generation_history.jsonl" \
       "$DATA_DIR/generation_history_$FIRST_MONTH.jsonl"

    echo "✅ Rotated to: generation_history_$FIRST_MONTH.jsonl"

    # Count entries
    LINE_COUNT=$(wc -l < "$DATA_DIR/generation_history_$FIRST_MONTH.jsonl")
    FILE_SIZE=$(du -h "$DATA_DIR/generation_history_$FIRST_MONTH.jsonl" | cut -f1)
    echo "   Entries: $LINE_COUNT"
    echo "   Size: $FILE_SIZE"
    echo ""

    # Compress old archives (older than 3 months)
    echo "🗜️  Compressing archives older than 3 months..."
    COMPRESSED=0
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            gzip "$file"
            COMPRESSED=$((COMPRESSED + 1))
            echo "   Compressed: $(basename "$file")"
        fi
    done < <(find "$DATA_DIR" -name "generation_history_*.jsonl" -mtime +90)

    if [ $COMPRESSED -eq 0 ]; then
        echo "   No files to compress"
    else
        echo "   Compressed $COMPRESSED files"
    fi
    echo ""

    # Move compressed archives (older than 6 months) to archive dir
    mkdir -p "$ARCHIVE_DIR"
    echo "📦 Archiving files older than 6 months..."
    ARCHIVED=0
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            mv "$file" "$ARCHIVE_DIR/"
            ARCHIVED=$((ARCHIVED + 1))
            echo "   Archived: $(basename "$file")"
        fi
    done < <(find "$DATA_DIR" -name "generation_history_*.jsonl.gz" -mtime +180)

    if [ $ARCHIVED -eq 0 ]; then
        echo "   No files to archive"
    else
        echo "   Archived $ARCHIVED files to: $ARCHIVE_DIR"
    fi
    echo ""

    echo "✅ Log rotation complete!"
else
    echo "ℹ️  Current log is from this month - no rotation needed"
fi

echo ""
echo "Current data directory status:"
echo "========================================"
ls -lh "$DATA_DIR"/*.jsonl 2>/dev/null || echo "No JSONL files"
echo ""
echo "Archive directory:"
ls -lh "$ARCHIVE_DIR"/*.gz 2>/dev/null || echo "No archived files"
echo ""
echo "Storage usage:"
du -sh "$DATA_DIR"
echo ""
