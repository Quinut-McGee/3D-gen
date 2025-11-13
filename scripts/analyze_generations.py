#!/usr/bin/env python3
"""
Analyze generation history to identify patterns and bottlenecks.

Usage:
    # Overall statistics
    python scripts/analyze_generations.py --summary

    # Acceptance rate over time
    python scripts/analyze_generations.py --acceptance-rate --last-days 7

    # Failure analysis
    python scripts/analyze_generations.py --failures --task-type IMAGE-TO-3D

    # CLIP score distribution
    python scripts/analyze_generations.py --clip-distribution

    # Validator-specific analysis
    python scripts/analyze_generations.py --validator 57

    # Export for fine-tuning
    python scripts/analyze_generations.py --export-accepted --min-score 0.7
"""

import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import statistics


def load_generations(data_dir: Path, last_days: int = None):
    """Load generation history"""
    history_file = data_dir / "generation_history.jsonl"

    if not history_file.exists():
        print(f"❌ No generation history found: {history_file}")
        return []

    generations = []
    cutoff_time = None

    if last_days:
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=last_days)

    with open(history_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())

                # Filter by time if specified
                if cutoff_time:
                    timestamp = datetime.fromisoformat(entry['metadata']['timestamp'].replace('Z', '+00:00'))
                    if timestamp < cutoff_time:
                        continue

                generations.append(entry)
            except json.JSONDecodeError:
                continue

    return generations


def print_summary(generations):
    """Print summary statistics"""
    if not generations:
        print("No generations found")
        return

    total = len(generations)

    # Count by task type
    text_to_3d = sum(1 for g in generations if g['task']['task_type'] == 'TEXT-TO-3D')
    image_to_3d = sum(1 for g in generations if g['task']['task_type'] == 'IMAGE-TO-3D')

    # Count accepted
    accepted = sum(1 for g in generations if g['validator_feedback'].get('accepted'))

    # Count failures
    failed = sum(1 for g in generations if g['failure_analysis'].get('failed'))

    # Count pre-submission rejections
    pre_rejected = sum(1 for g in generations if g['generation']['submission'].get('pre_submission_rejection'))

    # Average CLIP scores
    clip_scores = [
        g['generation']['output'].get('quality_metrics', {}).get('clip_score')
        for g in generations
        if g['generation']['output'].get('quality_metrics', {}).get('clip_score') is not None
    ]

    # Average timing
    total_times = [
        g['generation']['timing'].get('total_time')
        for g in generations
        if g['generation']['timing'].get('total_time') is not None
    ]

    print("=" * 80)
    print("GENERATION HISTORY SUMMARY")
    print("=" * 80)
    print()
    print(f"Total generations: {total}")
    print()
    print("Task type breakdown:")
    print(f"  TEXT-TO-3D:  {text_to_3d} ({text_to_3d/total*100:.1f}%)")
    print(f"  IMAGE-TO-3D: {image_to_3d} ({image_to_3d/total*100:.1f}%)")
    print()
    print("Submission & Acceptance:")
    print(f"  Pre-submission rejections: {pre_rejected} ({pre_rejected/total*100:.1f}%)")
    print(f"  Submitted: {total - pre_rejected - failed} ({(total - pre_rejected - failed)/total*100:.1f}%)")
    print(f"  Validator accepted: {accepted} ({accepted/total*100:.1f}%)")
    print(f"  Failed: {failed} ({failed/total*100:.1f}%)")
    print()
    print("Quality Metrics:")
    if clip_scores:
        print(f"  Average CLIP: {statistics.mean(clip_scores):.4f}")
        print(f"  Median CLIP:  {statistics.median(clip_scores):.4f}")
        print(f"  CLIP range:   {min(clip_scores):.4f} - {max(clip_scores):.4f}")
    else:
        print("  No CLIP scores available")
    print()
    print("Timing:")
    if total_times:
        print(f"  Average generation time: {statistics.mean(total_times):.1f}s")
        print(f"  Median generation time:  {statistics.median(total_times):.1f}s")
        print(f"  Timing range: {min(total_times):.1f}s - {max(total_times):.1f}s")
    else:
        print("  No timing data available")


def print_acceptance_rate(generations, last_days=None):
    """Print acceptance rate analysis"""
    if not generations:
        print("No generations found")
        return

    submitted = [g for g in generations if g['generation']['submission'].get('submitted')]
    accepted = [g for g in submitted if g['validator_feedback'].get('accepted')]

    time_filter = f" (last {last_days} days)" if last_days else ""

    print("=" * 80)
    print(f"ACCEPTANCE RATE ANALYSIS{time_filter}")
    print("=" * 80)
    print()
    print(f"Total generations: {len(generations)}")
    print(f"Submitted: {len(submitted)} ({len(submitted)/len(generations)*100:.1f}%)")
    print(f"Accepted by validators: {len(accepted)} ({len(accepted)/len(submitted)*100:.1f}% of submitted)")
    print()
    print(f"Overall acceptance rate: {len(accepted)/len(generations)*100:.1f}% ({len(accepted)}/{len(generations)})")
    print()

    # Break down by task type
    for task_type in ['TEXT-TO-3D', 'IMAGE-TO-3D']:
        task_gens = [g for g in generations if g['task']['task_type'] == task_type]
        if not task_gens:
            continue

        task_submitted = [g for g in task_gens if g['generation']['submission'].get('submitted')]
        task_accepted = [g for g in task_submitted if g['validator_feedback'].get('accepted')]

        print(f"{task_type}:")
        print(f"  Total: {len(task_gens)}")
        print(f"  Submitted: {len(task_submitted)} ({len(task_submitted)/len(task_gens)*100:.1f}%)")
        if task_submitted:
            print(f"  Accepted: {len(task_accepted)} ({len(task_accepted)/len(task_submitted)*100:.1f}% of submitted)")
        print()


def print_failures(generations, task_type=None):
    """Print failure analysis"""
    failed = [g for g in generations if g['failure_analysis'].get('failed')]

    if task_type:
        failed = [g for g in failed if g['task']['task_type'] == task_type]

    if not failed:
        print(f"No failures found{' for ' + task_type if task_type else ''}")
        return

    # Count by error type
    error_types = defaultdict(int)
    for g in failed:
        error_type = g['failure_analysis'].get('error_type', 'unknown')
        error_types[error_type] += 1

    # Count pre-submission rejections
    pre_rejected = [g for g in generations if g['generation']['submission'].get('pre_submission_rejection')]
    rejection_reasons = defaultdict(int)
    for g in pre_rejected:
        reason = g['generation']['submission'].get('rejection_reason', 'unknown')
        rejection_reasons[reason] += 1

    print("=" * 80)
    print(f"FAILURE ANALYSIS{' - ' + task_type if task_type else ''}")
    print("=" * 80)
    print()
    print(f"Total failures: {len(failed)} ({len(failed)/len(generations)*100:.1f}%)")
    print()
    print("Failure types:")
    for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {error_type}: {count} ({count/len(failed)*100:.1f}%)")
    print()
    print(f"Pre-submission rejections: {len(pre_rejected)} ({len(pre_rejected)/len(generations)*100:.1f}%)")
    print()
    print("Rejection reasons:")
    for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count} ({count/len(pre_rejected)*100:.1f}%)")


def print_clip_distribution(generations):
    """Print CLIP score distribution"""
    clip_scores = [
        (g['task']['task_type'], g['generation']['output'].get('quality_metrics', {}).get('clip_score'))
        for g in generations
        if g['generation']['output'].get('quality_metrics', {}).get('clip_score') is not None
    ]

    if not clip_scores:
        print("No CLIP scores available")
        return

    print("=" * 80)
    print("CLIP SCORE DISTRIBUTION")
    print("=" * 80)
    print()

    # Overall distribution
    all_scores = [score for _, score in clip_scores]
    print(f"Total generations with CLIP scores: {len(all_scores)}")
    print(f"Mean: {statistics.mean(all_scores):.4f}")
    print(f"Median: {statistics.median(all_scores):.4f}")
    print(f"Std Dev: {statistics.stdev(all_scores):.4f}" if len(all_scores) > 1 else "Std Dev: N/A")
    print()

    # Distribution by task type
    for task_type in ['TEXT-TO-3D', 'IMAGE-TO-3D']:
        task_scores = [score for t, score in clip_scores if t == task_type]
        if not task_scores:
            continue

        print(f"{task_type}:")
        print(f"  Count: {len(task_scores)}")
        print(f"  Mean: {statistics.mean(task_scores):.4f}")
        print(f"  Median: {statistics.median(task_scores):.4f}")
        print(f"  Range: {min(task_scores):.4f} - {max(task_scores):.4f}")

        # Histogram
        bins = [(0, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 1.0)]
        print("  Distribution:")
        for low, high in bins:
            count = sum(1 for s in task_scores if low <= s < high)
            pct = count / len(task_scores) * 100
            bar = "█" * int(pct / 2)
            print(f"    {low:.2f}-{high:.2f}: {count:3d} ({pct:4.1f}%) {bar}")
        print()


def print_validator_analysis(generations, validator_uid):
    """Print validator-specific analysis"""
    validator_gens = [g for g in generations if g['metadata']['validator_uid'] == validator_uid]

    if not validator_gens:
        print(f"No generations found for validator {validator_uid}")
        return

    print("=" * 80)
    print(f"VALIDATOR {validator_uid} ANALYSIS")
    print("=" * 80)
    print()
    print(f"Total generations: {len(validator_gens)}")
    print()

    # Acceptance rate
    submitted = [g for g in validator_gens if g['generation']['submission'].get('submitted')]
    accepted = [g for g in submitted if g['validator_feedback'].get('accepted')]

    print(f"Submitted: {len(submitted)}")
    if submitted:
        print(f"Accepted: {len(accepted)} ({len(accepted)/len(submitted)*100:.1f}%)")
    print()

    # CLIP scores
    clip_scores = [
        g['generation']['output'].get('quality_metrics', {}).get('clip_score')
        for g in validator_gens
        if g['generation']['output'].get('quality_metrics', {}).get('clip_score') is not None
    ]

    if clip_scores:
        print("CLIP scores:")
        print(f"  Mean: {statistics.mean(clip_scores):.4f}")
        print(f"  Median: {statistics.median(clip_scores):.4f}")
        print()

    # Gaussian counts for accepted generations
    accepted_gaussian_counts = [
        g['generation']['output'].get('num_gaussians')
        for g in accepted
        if g['generation']['output'].get('num_gaussians') is not None
    ]

    if accepted_gaussian_counts:
        print("Accepted generations - Gaussian counts:")
        print(f"  Mean: {statistics.mean(accepted_gaussian_counts):,.0f}")
        print(f"  Median: {statistics.median(accepted_gaussian_counts):,.0f}")
        print(f"  Min: {min(accepted_gaussian_counts):,.0f}")
        print()

    # Validator scores
    validator_scores = [
        g['validator_feedback'].get('score')
        for g in validator_gens
        if g['validator_feedback'].get('score') is not None
    ]

    if validator_scores:
        print("Validator scores:")
        print(f"  Mean: {statistics.mean(validator_scores):.4f}")
        print(f"  Median: {statistics.median(validator_scores):.4f}")
        print(f"  Range: {min(validator_scores):.4f} - {max(validator_scores):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze generation history")
    parser.add_argument('--data-dir', default='/home/kobe/404-gen/v1/3D-gen/data', help='Data directory')
    parser.add_argument('--summary', action='store_true', help='Print summary statistics')
    parser.add_argument('--acceptance-rate', action='store_true', help='Print acceptance rate analysis')
    parser.add_argument('--failures', action='store_true', help='Print failure analysis')
    parser.add_argument('--clip-distribution', action='store_true', help='Print CLIP score distribution')
    parser.add_argument('--validator', type=int, help='Analyze specific validator')
    parser.add_argument('--task-type', choices=['TEXT-TO-3D', 'IMAGE-TO-3D'], help='Filter by task type')
    parser.add_argument('--last-days', type=int, help='Only analyze last N days')

    args = parser.parse_args()

    # Load generations
    data_dir = Path(args.data_dir)
    generations = load_generations(data_dir, args.last_days)

    if not generations:
        print("No generation history found")
        return

    # Run requested analysis
    if args.summary:
        print_summary(generations)
    elif args.acceptance_rate:
        print_acceptance_rate(generations, args.last_days)
    elif args.failures:
        print_failures(generations, args.task_type)
    elif args.clip_distribution:
        print_clip_distribution(generations)
    elif args.validator:
        print_validator_analysis(generations, args.validator)
    else:
        # Default: show summary
        print_summary(generations)


if __name__ == "__main__":
    main()
