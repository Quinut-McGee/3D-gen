#!/usr/bin/env python3
"""
Correlate Validator Feedback with Generation Metrics - GROUND TRUTH ANALYSIS

This script mines actual validator decisions from PM2 logs and correlates them
with generation metrics to identify what ACTUALLY causes Score=0.0.

NO assumptions. NO theoretical metrics. Just data.
"""

import re
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import statistics


class ValidatorCorrelator:
    """Correlate validator feedback with generation metrics using real data"""

    def __init__(self):
        # Check archived log first (has historical data)
        archived_log = Path("/home/kobe/.pm2/logs/miner-sn17-mainnet.log")
        current_log = Path("/home/kobe/.pm2/logs/miner-sn17-mainnet-out.log")

        if archived_log.exists() and archived_log.stat().st_size > 1_000_000:
            self.miner_log = archived_log
        else:
            self.miner_log = current_log

        # Check archived worker log too
        archived_gen = Path("/home/kobe/.pm2/logs/gen-worker-1.log")
        current_gen = Path("/home/kobe/.pm2/logs/gen-worker-1-error.log")

        if archived_gen.exists() and archived_gen.stat().st_size > 100_000:
            self.gen_log = archived_gen
        else:
            self.gen_log = current_gen

    def parse_validator_feedback(self) -> List[Dict]:
        """Extract all validator feedback from miner logs"""
        feedback = []

        # Pattern from actual logs: [2025-10-30 17:16:49.701] | INFO | ... | Feedback from [49]: Score=0.7503282427787781
        # Also handles Score=0.0 and Score=failed
        pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\].*Feedback from \[(\d+)\]: Score=([\d.]+|failed)'

        # Use grep -a to handle binary file
        import subprocess
        result = subprocess.run(
            ['grep', '-a', 'Feedback from \\[', str(self.miner_log)],
            capture_output=True,
            text=True
        )

        for line in result.stdout.split('\n'):
            if not line:
                continue

            match = re.search(pattern, line)
            if match:
                timestamp_str, validator_uid, score_str = match.groups()
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')

                # Skip "failed" scores (network errors, not quality issues)
                if score_str == 'failed':
                    continue

                feedback.append({
                    'timestamp': timestamp,
                    'validator_uid': int(validator_uid),
                    'score': float(score_str),
                    'raw_line': line.strip()
                })

        return sorted(feedback, key=lambda x: x['timestamp'])

    def parse_generation_metrics(self) -> List[Dict]:
        """Extract generation metrics from worker logs"""
        generations = []

        # Look for generation completion messages with metrics
        # Pattern: [timestamp] ✅ Successfully generated prompt (gaussians: X, size: Y MB, time: Z s)

        with open(self.gen_log, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            # Look for successful generation completion
            if '✅ Successfully generated' in line or '✓ Generation successful' in line:
                # Extract timestamp
                timestamp_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if not timestamp_match:
                    continue

                timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')

                # Look backwards for prompt
                prompt = None
                for j in range(max(0, i-10), i):
                    if 'Processing text-to-3d request' in lines[j] or 'prompt' in lines[j].lower():
                        prompt_match = re.search(r'["\']([^"\']+)["\']', lines[j])
                        if prompt_match:
                            prompt = prompt_match.group(1)
                            break

                # Look backwards and forwards for metrics
                gaussian_count = None
                file_size_mb = None
                generation_time = None
                clip_score = None

                # Search surrounding lines (±5 lines)
                for j in range(max(0, i-5), min(len(lines), i+5)):
                    search_line = lines[j]

                    # Gaussian count
                    gauss_match = re.search(r'num_gaussians[:\s=]+(\d+)|gaussians?[:\s=]+(\d[\d,]+)', search_line, re.IGNORECASE)
                    if gauss_match and not gaussian_count:
                        count_str = gauss_match.group(1) or gauss_match.group(2)
                        gaussian_count = int(count_str.replace(',', ''))

                    # File size
                    size_match = re.search(r'size[:\s=]+([\d.]+)\s*MB|file.*?([\d.]+)\s*MB', search_line, re.IGNORECASE)
                    if size_match and not file_size_mb:
                        file_size_mb = float(size_match.group(1) or size_match.group(2))

                    # Generation time
                    time_match = re.search(r'time[:\s=]+([\d.]+)\s*s|took\s+([\d.]+)\s*s', search_line, re.IGNORECASE)
                    if time_match and not generation_time:
                        generation_time = float(time_match.group(1) or time_match.group(2))

                    # CLIP score
                    clip_match = re.search(r'CLIP.*score[:\s=]+([\d.]+)|validation.*score[:\s=]+([\d.]+)', search_line, re.IGNORECASE)
                    if clip_match and not clip_score:
                        clip_score = float(clip_match.group(1) or clip_match.group(2))

                if gaussian_count:  # Only add if we found gaussian count
                    generations.append({
                        'timestamp': timestamp,
                        'prompt': prompt or 'unknown',
                        'gaussian_count': gaussian_count,
                        'file_size_mb': file_size_mb,
                        'generation_time': generation_time,
                        'clip_score': clip_score,
                    })

        return sorted(generations, key=lambda x: x['timestamp'])

    def correlate(self, generations: List[Dict], feedback: List[Dict],
                  window_seconds: int = 60) -> List[Dict]:
        """
        Match generations with validator feedback based on timestamps.

        A generation at time T should receive validator feedback within T+30s to T+60s
        """
        correlated = []

        for gen in generations:
            gen_time = gen['timestamp']

            # Look for validator feedback within window
            matching_feedback = []
            for fb in feedback:
                fb_time = fb['timestamp']
                time_diff = (fb_time - gen_time).total_seconds()

                # Validator responds 20-60 seconds after generation
                if 20 <= time_diff <= window_seconds:
                    matching_feedback.append({
                        'validator_uid': fb['validator_uid'],
                        'score': fb['score'],
                        'time_diff': time_diff
                    })

            if matching_feedback:
                # Take the feedback closest in time
                closest = min(matching_feedback, key=lambda x: x['time_diff'])

                correlated.append({
                    **gen,  # All generation metrics
                    'validator_uid': closest['validator_uid'],
                    'validator_score': closest['score'],
                    'feedback_delay': closest['time_diff']
                })

        return correlated

    def analyze_patterns(self, correlated: List[Dict]):
        """
        Find the ACTUAL differences between Score=0.0 and Score>0.0 generations.

        This is the ground truth. No assumptions.
        """
        if not correlated:
            print("❌ No correlated data found. Cannot analyze patterns.")
            return

        # Split by score
        score_zero = [c for c in correlated if c['validator_score'] == 0.0]
        score_nonzero = [c for c in correlated if c['validator_score'] > 0.0]

        print("=" * 80)
        print("VALIDATOR FEEDBACK CORRELATION ANALYSIS - GROUND TRUTH")
        print("=" * 80)
        print(f"\nTotal correlated generations: {len(correlated)}")
        print(f"  Score = 0.0:  {len(score_zero)} ({len(score_zero)/len(correlated)*100:.1f}%)")
        print(f"  Score > 0.0:  {len(score_nonzero)} ({len(score_nonzero)/len(correlated)*100:.1f}%)")

        if not score_zero or not score_nonzero:
            print("\n⚠️  Need both Score=0.0 and Score>0.0 samples for comparison")
            return

        print("\n" + "=" * 80)
        print("METRIC COMPARISON: What ACTUALLY Differentiates Score=0.0 vs Score>0.0")
        print("=" * 80)

        # Compare gaussian counts
        zero_gaussians = [c['gaussian_count'] for c in score_zero if c['gaussian_count']]
        nonzero_gaussians = [c['gaussian_count'] for c in score_nonzero if c['gaussian_count']]

        if zero_gaussians and nonzero_gaussians:
            print(f"\n📊 Gaussian Count:")
            print(f"   Score=0.0 average:  {statistics.mean(zero_gaussians):,.0f}")
            print(f"   Score>0.0 average:  {statistics.mean(nonzero_gaussians):,.0f}")
            print(f"   Score=0.0 median:   {statistics.median(zero_gaussians):,.0f}")
            print(f"   Score>0.0 median:   {statistics.median(nonzero_gaussians):,.0f}")

            diff_pct = ((statistics.mean(nonzero_gaussians) - statistics.mean(zero_gaussians)) /
                       statistics.mean(zero_gaussians) * 100)
            print(f"   Difference: {diff_pct:+.1f}%")

            if abs(diff_pct) > 30:
                print(f"   🔴 MAJOR FACTOR - This is likely critical!")
            elif abs(diff_pct) > 15:
                print(f"   🟡 SIGNIFICANT FACTOR")
            else:
                print(f"   ✅ Not a major differentiator")

        # Compare file sizes
        zero_sizes = [c['file_size_mb'] for c in score_zero if c['file_size_mb']]
        nonzero_sizes = [c['file_size_mb'] for c in score_nonzero if c['file_size_mb']]

        if zero_sizes and nonzero_sizes:
            print(f"\n📦 File Size (MB):")
            print(f"   Score=0.0 average:  {statistics.mean(zero_sizes):.1f} MB")
            print(f"   Score>0.0 average:  {statistics.mean(nonzero_sizes):.1f} MB")

            diff_pct = ((statistics.mean(nonzero_sizes) - statistics.mean(zero_sizes)) /
                       statistics.mean(zero_sizes) * 100)
            print(f"   Difference: {diff_pct:+.1f}%")

            if abs(diff_pct) > 30:
                print(f"   🔴 MAJOR FACTOR")
            elif abs(diff_pct) > 15:
                print(f"   🟡 SIGNIFICANT FACTOR")
            else:
                print(f"   ✅ Not a major differentiator")

        # Compare generation times
        zero_times = [c['generation_time'] for c in score_zero if c['generation_time']]
        nonzero_times = [c['generation_time'] for c in score_nonzero if c['generation_time']]

        if zero_times and nonzero_times:
            print(f"\n⏱️  Generation Time (seconds):")
            print(f"   Score=0.0 average:  {statistics.mean(zero_times):.1f}s")
            print(f"   Score>0.0 average:  {statistics.mean(nonzero_times):.1f}s")

            diff_pct = ((statistics.mean(nonzero_times) - statistics.mean(zero_times)) /
                       statistics.mean(zero_times) * 100)
            print(f"   Difference: {diff_pct:+.1f}%")

        # Compare CLIP scores (if available)
        zero_clip = [c['clip_score'] for c in score_zero if c['clip_score']]
        nonzero_clip = [c['clip_score'] for c in score_nonzero if c['clip_score']]

        if zero_clip and nonzero_clip:
            print(f"\n🎯 CLIP Score (2D image quality):")
            print(f"   Score=0.0 average:  {statistics.mean(zero_clip):.3f}")
            print(f"   Score>0.0 average:  {statistics.mean(nonzero_clip):.3f}")

            diff_pct = ((statistics.mean(nonzero_clip) - statistics.mean(zero_clip)) /
                       statistics.mean(zero_clip) * 100)
            print(f"   Difference: {diff_pct:+.1f}%")

            if abs(diff_pct) > 30:
                print(f"   🔴 MAJOR FACTOR - Image quality is critical!")
            elif abs(diff_pct) > 15:
                print(f"   🟡 SIGNIFICANT FACTOR")

        # Distribution analysis
        print("\n" + "=" * 80)
        print("DISTRIBUTION ANALYSIS")
        print("=" * 80)

        # Gaussian count bins
        print(f"\n📊 Gaussian Count Distribution:")
        bins = [0, 100000, 150000, 200000, 300000, 500000, float('inf')]
        bin_labels = ['<100K', '100-150K', '150-200K', '200-300K', '300-500K', '>500K']

        for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            zero_in_bin = sum(1 for g in zero_gaussians if low <= g < high)
            nonzero_in_bin = sum(1 for g in nonzero_gaussians if low <= g < high)

            zero_pct = zero_in_bin / len(zero_gaussians) * 100 if zero_gaussians else 0
            nonzero_pct = nonzero_in_bin / len(nonzero_gaussians) * 100 if nonzero_gaussians else 0

            print(f"   {bin_labels[i]:>12}: Score=0.0: {zero_pct:5.1f}%  |  Score>0.0: {nonzero_pct:5.1f}%")

        # Sample some Score=0.0 cases
        print("\n" + "=" * 80)
        print("SAMPLE Score=0.0 GENERATIONS (for manual inspection)")
        print("=" * 80)

        for i, case in enumerate(score_zero[:5]):
            print(f"\nCase {i+1}:")
            print(f"  Prompt: {case['prompt'][:60]}...")
            print(f"  Gaussians: {case['gaussian_count']:,}")
            print(f"  File size: {case['file_size_mb']:.1f} MB" if case['file_size_mb'] else "  File size: N/A")
            print(f"  Validator: UID {case['validator_uid']}")
            print(f"  Time: {case['timestamp']}")

        print("\n" + "=" * 80)
        print("ACTIONABLE FINDINGS")
        print("=" * 80)
        print("\nBased on ACTUAL validator behavior (not assumptions):")

        # Determine root cause
        if zero_gaussians and nonzero_gaussians:
            gaussian_diff = abs((statistics.mean(nonzero_gaussians) - statistics.mean(zero_gaussians)) /
                              statistics.mean(zero_gaussians) * 100)

            if gaussian_diff > 30:
                print("\n🔴 PRIMARY ISSUE: Gaussian Count")
                print(f"   Score=0.0 generations have {gaussian_diff:.0f}% fewer gaussians")
                print(f"   Action: Increase TRELLIS sparse_structure steps or CFG strength")
            elif statistics.mean(zero_gaussians) < 150000:
                print("\n🔴 PRIMARY ISSUE: Sparse Models")
                print(f"   Many Score=0.0 generations below 150K gaussian threshold")
                print(f"   Action: Reject generations below threshold before submission")

        if zero_clip and nonzero_clip:
            clip_diff = abs((statistics.mean(nonzero_clip) - statistics.mean(zero_clip)) /
                          statistics.mean(zero_clip) * 100)

            if clip_diff > 30:
                print("\n🔴 PRIMARY ISSUE: Image Quality (CLIP Score)")
                print(f"   Score=0.0 generations have {clip_diff:.0f}% lower CLIP scores")
                print(f"   Action: Fix SD3.5 image generation quality")

        print("\n" + "=" * 80)


def main():
    correlator = ValidatorCorrelator()

    print("Mining historical validator feedback and generation metrics...")
    print("This uses REAL data - no assumptions.\n")

    # Parse both logs
    feedback = correlator.parse_validator_feedback()
    print(f"✓ Extracted {len(feedback)} validator feedback events")

    generations = correlator.parse_generation_metrics()
    print(f"✓ Extracted {len(generations)} generation events with metrics")

    # Correlate them
    correlated = correlator.correlate(generations, feedback, window_seconds=60)
    print(f"✓ Correlated {len(correlated)} generation→validator pairs\n")

    # Analyze patterns
    correlator.analyze_patterns(correlated)

    # Export correlated data for further analysis
    output_path = Path("/tmp/correlated_validator_feedback.jsonl")
    with open(output_path, 'w') as f:
        for record in correlated:
            # Convert datetime to string for JSON serialization
            record_copy = record.copy()
            record_copy['timestamp'] = record['timestamp'].isoformat()
            f.write(json.dumps(record_copy) + '\n')

    print(f"\n💾 Exported correlated data to: {output_path}")
    print(f"   Use this for further analysis or visualization")


if __name__ == "__main__":
    main()
