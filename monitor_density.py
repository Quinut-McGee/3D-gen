#!/usr/bin/env python3
"""
Monitor Gaussian density distribution for Option A+B deployment
Tracks improvement from 337K avg (30% high-density) baseline
"""
import os
import re
from datetime import datetime

def analyze_gaussian_density(log_file, num_samples=50):
    """Extract and analyze gaussian counts from generation logs"""
    gaussian_counts = []

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        # Search for generation stats in recent lines
        for line in lines[-2000:]:
            if '📊 Generation stats:' in line:
                # Extract: "📊 Generation stats: 154,336 gaussians, 10.0MB"
                match = re.search(r'(\d{1,3}(?:,\d{3})*)\s+gaussians', line)
                if match:
                    count_str = match.group(1).replace(',', '')
                    gaussian_counts.append(int(count_str))

    except FileNotFoundError:
        print(f"❌ Error: Log file not found: {log_file}")
        return None

    if not gaussian_counts:
        print("⚠️  No gaussian count data found in recent logs")
        return None

    # Analyze last N submissions
    recent = gaussian_counts[-num_samples:] if len(gaussian_counts) >= num_samples else gaussian_counts

    # Calculate metrics
    avg = sum(recent) / len(recent)
    high = sum(1 for c in recent if c > 400000)
    med = sum(1 for c in recent if 150000 <= c <= 400000)
    low = sum(1 for c in recent if c < 150000)

    return {
        'total_samples': len(recent),
        'average': int(avg),
        'high_density_count': high,
        'high_density_pct': 100 * high / len(recent),
        'med_density_count': med,
        'med_density_pct': 100 * med / len(recent),
        'low_density_count': low,
        'low_density_pct': 100 * low / len(recent),
        'min': min(recent),
        'max': max(recent),
        'all_counts': recent
    }


def print_report(stats, baseline_avg=337579, baseline_high_pct=30):
    """Print formatted analysis report"""
    print(f"\n{'='*60}")
    print(f"GAUSSIAN DENSITY ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"\n📊 Distribution (Last {stats['total_samples']} submissions):")
    print(f"   High (>400K):  {stats['high_density_count']:2d} submissions ({stats['high_density_pct']:5.1f}%)")
    print(f"   Med (150-400K): {stats['med_density_count']:2d} submissions ({stats['med_density_pct']:5.1f}%)")
    print(f"   Low (<150K):   {stats['low_density_count']:2d} submissions ({stats['low_density_pct']:5.1f}%)")

    print(f"\n📈 Statistics:")
    print(f"   Average:  {stats['average']:,} gaussians")
    print(f"   Range:    {stats['min']:,} → {stats['max']:,}")

    print(f"\n🎯 Improvement vs Baseline:")
    avg_delta = stats['average'] - baseline_avg
    avg_delta_pct = 100 * avg_delta / baseline_avg
    high_delta = stats['high_density_pct'] - baseline_high_pct

    avg_symbol = "✅" if avg_delta > 0 else "❌"
    high_symbol = "✅" if high_delta > 0 else "❌"

    print(f"   Average:      {avg_symbol} {avg_delta:+,} gaussians ({avg_delta_pct:+.1f}%)")
    print(f"   High-density: {high_symbol} {high_delta:+.1f}% (from {baseline_high_pct}% → {stats['high_density_pct']:.1f}%)")

    print(f"\n📌 Target Metrics:")
    print(f"   Average:      {'✅' if stats['average'] >= 400000 else '⏳'} {stats['average']:,} / 400,000+ target")
    print(f"   High-density: {'✅' if stats['high_density_pct'] >= 50 else '⏳'} {stats['high_density_pct']:.1f}% / 50%+ target")
    print(f"   Low-density:  {'✅' if stats['low_density_pct'] <= 25 else '⏳'} {stats['low_density_pct']:.1f}% / <25% target")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import sys

    log_file = "/home/kobe/.pm2/logs/gen-worker-1-error.log"
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    stats = analyze_gaussian_density(log_file, num_samples)

    if stats:
        print_report(stats)

        # Return exit code based on targets
        if stats['average'] >= 400000 and stats['high_density_pct'] >= 50:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Targets not met yet
    else:
        sys.exit(2)  # Error reading data
