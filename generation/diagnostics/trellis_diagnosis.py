#!/usr/bin/env python3
"""
Diagnose TRELLIS rejection root cause by analyzing validator feedback patterns
"""
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict

def analyze_trellis_rejections():
    miner_log = Path("/home/kobe/.pm2/logs/miner-sn17-mainnet.log")

    print("="*80)
    print("TRELLIS ERA VALIDATOR FEEDBACK ANALYSIS")
    print("="*80)

    # Extract TRELLIS-era feedback (Nov 1, 01:00-03:30)
    feedback_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Feedback from \[(\d+)\]: Score=([\d.]+|failed)'

    feedback_events = []
    with open(miner_log, 'r', errors='ignore') as f:
        for line in f:
            # Filter to TRELLIS period
            if not ('2025-11-01 01:' in line or '2025-11-01 02:' in line or '2025-11-01 03:' in line[:20]):
                continue

            match = re.search(feedback_pattern, line)
            if match:
                ts, validator, score = match.groups()
                if score != 'failed':
                    feedback_events.append({
                        'timestamp': ts,
                        'validator': int(validator),
                        'score': float(score),
                        'raw_line': line.strip()
                    })

    if not feedback_events:
        print("\n❌ No TRELLIS-era feedback found!")
        return

    # Statistics
    total = len(feedback_events)
    zeros = sum(1 for f in feedback_events if f['score'] == 0.0)
    non_zeros = [f for f in feedback_events if f['score'] > 0.0]

    print(f"\nTotal submissions: {total}")
    print(f"Score=0.0: {zeros} ({zeros/total*100:.1f}%)")
    print(f"Score>0.0: {len(non_zeros)} ({len(non_zeros)/total*100:.1f}%)")

    if non_zeros:
        scores = [f['score'] for f in non_zeros]
        print(f"\nSuccessful scores:")
        print(f"  Range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"  Average: {sum(scores)/len(scores):.3f}")
        print(f"  Median: {sorted(scores)[len(scores)//2]:.3f}")

    # Per-validator analysis
    validator_stats = defaultdict(lambda: {'total': 0, 'zeros': 0})
    for f in feedback_events:
        vid = f['validator']
        validator_stats[vid]['total'] += 1
        if f['score'] == 0.0:
            validator_stats[vid]['zeros'] += 1

    print("\n" + "="*80)
    print("PER-VALIDATOR REJECTION RATES")
    print("="*80)
    for vid in sorted(validator_stats.keys()):
        stats = validator_stats[vid]
        reject_rate = stats['zeros'] / stats['total'] * 100
        status = "❌ HIGH" if reject_rate > 70 else "⚠️  MEDIUM" if reject_rate > 40 else "✅ LOW"
        print(f"{status} Validator {vid}: {stats['zeros']}/{stats['total']} = {reject_rate:.1f}% rejection")

    # Timeline analysis
    print("\n" + "="*80)
    print("TIMELINE ANALYSIS")
    print("="*80)

    hourly = defaultdict(lambda: {'total': 0, 'zeros': 0})
    for f in feedback_events:
        hour = f['timestamp'][11:13]  # Extract hour
        hourly[hour]['total'] += 1
        if f['score'] == 0.0:
            hourly[hour]['zeros'] += 1

    for hour in sorted(hourly.keys()):
        stats = hourly[hour]
        reject_rate = stats['zeros'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"Hour {hour}:00 - {stats['total']} submissions, {reject_rate:.0f}% rejected")

    # Sample successful vs failed submissions
    print("\n" + "="*80)
    print("SAMPLE SUCCESSFUL SUBMISSIONS (for comparison)")
    print("="*80)
    for f in non_zeros[:5]:
        print(f"  {f['timestamp']} | Validator {f['validator']} | Score {f['score']:.3f}")

    print("\n" + "="*80)
    print("SAMPLE REJECTED SUBMISSIONS")
    print("="*80)
    rejected = [f for f in feedback_events if f['score'] == 0.0]
    for f in rejected[:5]:
        print(f"  {f['timestamp']} | Validator {f['validator']} | Score 0.0")

if __name__ == "__main__":
    analyze_trellis_rejections()
