#!/usr/bin/env python3
"""
Export high-quality validator-accepted generations for fine-tuning.

Usage:
    # Export IMAGE-TO-3D accepted generations (score ≥0.7)
    python scripts/export_training_data.py \
        --task-type IMAGE-TO-3D \
        --min-score 0.7 \
        --output /home/kobe/404-gen/v1/3D-gen/data/training/image_to_3d_accepted.jsonl

    # Export TEXT-TO-3D prompts that led to high scores
    python scripts/export_training_data.py \
        --task-type TEXT-TO-3D \
        --min-score 0.8 \
        --output /home/kobe/404-gen/v1/3D-gen/data/training/text_to_3d_prompts.jsonl

    # Export all accepted with min CLIP score
    python scripts/export_training_data.py \
        --min-score 0.6 \
        --min-clip 0.20 \
        --output training_data.jsonl
"""

import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone


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


def filter_for_training(
    generations,
    task_type=None,
    min_score=None,
    min_clip=None,
    min_gaussians=None,
    only_accepted=True
):
    """
    Filter generations for training data.

    Args:
        generations: List of generation entries
        task_type: Filter by task type (TEXT-TO-3D or IMAGE-TO-3D)
        min_score: Minimum validator score (0-1)
        min_clip: Minimum CLIP score
        min_gaussians: Minimum gaussian count
        only_accepted: Only include validator-accepted (score > 0)

    Returns:
        Filtered list of generations
    """
    filtered = []

    for g in generations:
        # Filter by task type
        if task_type and g['task']['task_type'] != task_type:
            continue

        # Filter by acceptance
        if only_accepted and not g['validator_feedback'].get('accepted'):
            continue

        # Filter by validator score
        if min_score:
            score = g['validator_feedback'].get('score')
            if score is None or score < min_score:
                continue

        # Filter by CLIP score
        if min_clip:
            clip_score = g['generation']['output'].get('quality_metrics', {}).get('clip_score')
            if clip_score is None or clip_score < min_clip:
                continue

        # Filter by gaussian count
        if min_gaussians:
            num_gaussians = g['generation']['output'].get('num_gaussians')
            if num_gaussians is None or num_gaussians < min_gaussians:
                continue

        filtered.append(g)

    return filtered


def export_training_data(generations, output_path: Path):
    """
    Export training data in simplified format.

    Format:
    {
        "generation_id": "uuid",
        "task_type": "IMAGE-TO-3D",
        "prompt": "original prompt or [BASE64_IMAGE_HASH:...]",
        "enhanced_prompt": "LLM-enhanced prompt",
        "image_file_path": "images/abc123.png" (if IMAGE-TO-3D),
        "output_ply_path": "ply_files/uuid_hash.ply",
        "output_ply_hash": "sha256:...",
        "validator_score": 0.75,
        "clip_score": 0.234,
        "num_gaussians": 412356,
        "gaussian_stats": {...},
        "miner_config": {...},
        "timestamp": "2025-11-12T21:30:45Z"
    }
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exported_count = 0

    with open(output_path, 'w') as f:
        for g in generations:
            # Extract relevant training data
            training_entry = {
                "generation_id": g['metadata']['generation_id'],
                "task_type": g['task']['task_type'],
                "prompt": g['task']['prompt'],
                "enhanced_prompt": g['generation'].get('enhanced_prompt'),
                "image_file_path": g['task'].get('image_file_path'),
                "output_ply_path": g['generation']['output'].get('ply_file_path'),
                "output_ply_hash": g['generation']['output'].get('ply_file_hash'),
                "validator_score": g['validator_feedback'].get('score'),
                "clip_score": g['generation']['output'].get('quality_metrics', {}).get('clip_score'),
                "num_gaussians": g['generation']['output'].get('num_gaussians'),
                "gaussian_stats": g['generation']['output'].get('gaussian_stats'),
                "miner_config": g.get('miner_config'),
                "timestamp": g['metadata']['timestamp']
            }

            f.write(json.dumps(training_entry) + '\n')
            exported_count += 1

    return exported_count


def main():
    parser = argparse.ArgumentParser(description="Export training data from generation history")
    parser.add_argument('--data-dir', default='/home/kobe/404-gen/v1/3D-gen/data', help='Data directory')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--task-type', choices=['TEXT-TO-3D', 'IMAGE-TO-3D'], help='Filter by task type')
    parser.add_argument('--min-score', type=float, help='Minimum validator score (0-1)')
    parser.add_argument('--min-clip', type=float, help='Minimum CLIP score')
    parser.add_argument('--min-gaussians', type=int, help='Minimum gaussian count')
    parser.add_argument('--last-days', type=int, help='Only export from last N days')
    parser.add_argument('--include-rejected', action='store_true', help='Include validator-rejected generations')

    args = parser.parse_args()

    # Load generations
    data_dir = Path(args.data_dir)
    generations = load_generations(data_dir, args.last_days)

    if not generations:
        print("No generation history found")
        return

    print("=" * 80)
    print("EXPORT TRAINING DATA")
    print("=" * 80)
    print()
    print(f"Total generations loaded: {len(generations)}")
    print()

    # Filter for training
    filtered = filter_for_training(
        generations,
        task_type=args.task_type,
        min_score=args.min_score,
        min_clip=args.min_clip,
        min_gaussians=args.min_gaussians,
        only_accepted=not args.include_rejected
    )

    print(f"Generations matching criteria: {len(filtered)}")
    print()

    if args.task_type:
        print(f"  Task type: {args.task_type}")
    if args.min_score:
        print(f"  Min validator score: {args.min_score}")
    if args.min_clip:
        print(f"  Min CLIP score: {args.min_clip}")
    if args.min_gaussians:
        print(f"  Min gaussians: {args.min_gaussians:,}")
    if not args.include_rejected:
        print(f"  Only validator-accepted")
    print()

    if not filtered:
        print("❌ No generations match the specified criteria")
        return

    # Export
    output_path = Path(args.output)
    count = export_training_data(filtered, output_path)

    print(f"✅ Exported {count} training examples to: {output_path}")
    print()
    print("Training data format:")
    print("  • JSONL (one JSON object per line)")
    print("  • Each entry contains prompt, output, scores, and config")
    print("  • Image/PLY file paths included (actual files in data/ directory)")
    print()
    print("Next steps:")
    print("  1. Review exported data")
    print("  2. Prepare for fine-tuning (load images/PLY files as needed)")
    print("  3. Use for model improvement")


if __name__ == "__main__":
    main()
