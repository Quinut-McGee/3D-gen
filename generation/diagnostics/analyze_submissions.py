#!/usr/bin/env python3
"""
Analyze submission database to find patterns in Score=0.0 vs Score>0
"""
import json
import sys
from pathlib import Path
from collections import defaultdict


def load_database(db_path="/tmp/submission_database.jsonl"):
    """Load all submissions from database"""
    db = Path(db_path)
    if not db.exists():
        print(f"❌ Database not found: {db_path}")
        return []

    records = []
    with open(db, 'r') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return records


def analyze_patterns(records):
    """Find patterns that correlate with Score=0.0"""
    if not records:
        print("No records to analyze")
        return

    # Separate by validator score
    score_zero = [r for r in records if r.get('validator_score') == 0.0]
    score_good = [r for r in records if r.get('validator_score') is not None and r.get('validator_score') > 0.0]
    score_pending = [r for r in records if r.get('validator_score') is None]

    print("=" * 80)
    print("SUBMISSION DATABASE ANALYSIS")
    print("=" * 80)
    print(f"\nTotal submissions: {len(records)}")
    print(f"  Score = 0.0:  {len(score_zero)} ({len(score_zero)/len(records)*100:.1f}%)")
    print(f"  Score > 0.0:  {len(score_good)} ({len(score_good)/len(records)*100:.1f}%)")
    print(f"  Pending:      {len(score_pending)} ({len(score_pending)/len(records)*100:.1f}%)")

    if not score_zero and not score_good:
        print("\n⚠️  Not enough data yet - need validator feedback")
        return

    # Compare metrics
    print("\n" + "=" * 80)
    print("METRIC COMPARISON: Score=0.0 vs Score>0.0")
    print("=" * 80)

    metrics_to_compare = [
        ('gaussian_count', 'Gaussian Count'),
        ('file_size_mb', 'File Size (MB)'),
        ('generation_time', 'Generation Time (s)'),
    ]

    for key, name in metrics_to_compare:
        if score_zero and score_good:
            avg_zero = sum(r[key] for r in score_zero if key in r) / len(score_zero)
            avg_good = sum(r[key] for r in score_good if key in r) / len(score_good)
            diff_pct = ((avg_good - avg_zero) / avg_zero * 100) if avg_zero > 0 else 0

            print(f"\n{name}:")
            print(f"  Score=0.0: {avg_zero:,.2f}")
            print(f"  Score>0.0: {avg_good:,.2f}")
            print(f"  Difference: {diff_pct:+.1f}%")

            if abs(diff_pct) > 20:
                print(f"  🔴 SIGNIFICANT DIFFERENCE! This may be a key factor.")

    # PLY Quality Metrics
    print("\n" + "=" * 80)
    print("PLY QUALITY METRICS COMPARISON")
    print("=" * 80)

    ply_metrics = [
        ('spatial_variance', 'Spatial Variance'),
        ('bbox_volume', 'Bounding Box Volume'),
        ('avg_opacity', 'Average Opacity'),
        ('avg_scale', 'Average Scale'),
        ('density_variance', 'Density Variance'),
    ]

    for key, name in ply_metrics:
        zero_values = [r['ply_quality_metrics'].get(key, 0) for r in score_zero if r.get('ply_quality_metrics')]
        good_values = [r['ply_quality_metrics'].get(key, 0) for r in score_good if r.get('ply_quality_metrics')]

        if zero_values and good_values:
            avg_zero = sum(zero_values) / len(zero_values)
            avg_good = sum(good_values) / len(good_values)
            diff_pct = ((avg_good - avg_zero) / avg_zero * 100) if avg_zero > 0 else 0

            print(f"\n{name}:")
            print(f"  Score=0.0: {avg_zero:.6f}")
            print(f"  Score>0.0: {avg_good:.6f}")
            print(f"  Difference: {diff_pct:+.1f}%")

            if abs(diff_pct) > 30:
                print(f"  🔴 MAJOR DIFFERENCE! This is likely a critical factor.")
            elif abs(diff_pct) > 20:
                print(f"  🟡 SIGNIFICANT DIFFERENCE! This may contribute to rejections.")

    # Validator-specific patterns
    print("\n" + "=" * 80)
    print("VALIDATOR-SPECIFIC PATTERNS")
    print("=" * 80)

    validator_scores = defaultdict(list)
    for r in records:
        if r.get('validator_uid') is not None and r.get('validator_score') is not None:
            validator_scores[r['validator_uid']].append(r['validator_score'])

    if validator_scores:
        print("\nScore distribution by validator:")
        for uid, scores in sorted(validator_scores.items()):
            avg_score = sum(scores) / len(scores)
            zero_rate = sum(1 for s in scores if s == 0.0) / len(scores) * 100
            print(f"  Validator {uid}: avg={avg_score:.2f}, zero_rate={zero_rate:.1f}%, n={len(scores)}")

            if zero_rate > 80:
                print(f"    🔴 This validator rejects {zero_rate:.0f}% of submissions!")

    # Top issues detected
    print("\n" + "=" * 80)
    print("COMMON PLY QUALITY ISSUES")
    print("=" * 80)

    issue_counts = defaultdict(int)
    for r in score_zero:
        if r.get('ply_quality_metrics'):
            # Check for common issues
            metrics = r['ply_quality_metrics']
            if metrics.get('bbox_volume', 0) < 0.01:
                issue_counts['collapsed_model'] += 1
            if metrics.get('avg_opacity', 1.0) < 0.3:
                issue_counts['too_transparent'] += 1
            if metrics.get('spatial_variance', 0) < 0.1:
                issue_counts['clumped_gaussians'] += 1
            if metrics.get('avg_scale', 0) < 0.001:
                issue_counts['too_small'] += 1

    if issue_counts:
        print(f"\nIssues found in Score=0.0 submissions:")
        for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(score_zero) * 100
            print(f"  {issue.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
    else:
        print("\nNo obvious structural issues detected.")
        print("The problem may be visual quality rather than geometric.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/submission_database.jsonl"
    records = load_database(db_path)
    analyze_patterns(records)
