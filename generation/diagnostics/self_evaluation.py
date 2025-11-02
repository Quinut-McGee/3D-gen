#!/usr/bin/env python3
"""
Self-Evaluation Tool - Analyze generation quality independently of validators

This tool helps you understand what you're doing right and wrong by:
1. Scoring your own generations based on quality metrics
2. Comparing against historical best/worst cases
3. Identifying patterns in high-quality vs low-quality outputs
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import statistics


class SelfEvaluator:
    """Evaluate generation quality using internal metrics"""

    def __init__(self, db_path="/tmp/submission_database.jsonl"):
        self.db_path = Path(db_path)
        self.quality_weights = {
            # These weights define what makes a "good" generation
            # Adjust based on what you learn
            'gaussian_count': 0.15,        # More is generally better (up to a point)
            'spatial_variance': 0.25,      # Higher = better distribution
            'bbox_volume': 0.20,           # Proper size, not collapsed
            'avg_opacity': 0.15,           # Visible but not too opaque
            'avg_scale': 0.10,             # Proper gaussian size
            'density_uniformity': 0.15,    # Even distribution
        }

    def calculate_quality_score(self, metrics: Dict) -> float:
        """
        Calculate internal quality score (0-100) based on PLY metrics.
        This is YOUR score, independent of validators.
        """
        if not metrics.get('ply_quality_metrics'):
            return 0.0

        ply = metrics['ply_quality_metrics']
        score = 0.0

        # Gaussian count score (normalized to 150K-500K range)
        count = metrics.get('gaussian_count', 0)
        if count >= 150000:
            count_score = min((count - 150000) / (500000 - 150000), 1.0) * 100
        else:
            count_score = (count / 150000) * 50  # Penalty for below threshold
        score += count_score * self.quality_weights['gaussian_count']

        # Spatial variance score (higher is better, up to 1.0)
        spatial_var = ply.get('spatial_variance', 0)
        spatial_score = min(spatial_var / 1.0, 1.0) * 100
        score += spatial_score * self.quality_weights['spatial_variance']

        # Bounding box volume score (0.1-10.0 is good range)
        bbox_vol = ply.get('bbox_volume', 0)
        if 0.1 <= bbox_vol <= 10.0:
            bbox_score = 100
        elif bbox_vol < 0.1:
            bbox_score = (bbox_vol / 0.1) * 100
        else:
            bbox_score = max(0, 100 - (bbox_vol - 10.0) * 10)
        score += bbox_score * self.quality_weights['bbox_volume']

        # Opacity score (0.5-0.9 is ideal range)
        opacity = ply.get('avg_opacity', 0)
        if 0.5 <= opacity <= 0.9:
            opacity_score = 100
        elif opacity < 0.5:
            opacity_score = (opacity / 0.5) * 100
        else:
            opacity_score = max(0, 100 - (opacity - 0.9) * 200)
        score += opacity_score * self.quality_weights['avg_opacity']

        # Scale score (0.005-0.05 is good range)
        scale = ply.get('avg_scale', 0)
        if 0.005 <= scale <= 0.05:
            scale_score = 100
        elif scale < 0.005:
            scale_score = (scale / 0.005) * 100
        else:
            scale_score = max(0, 100 - (scale - 0.05) * 500)
        score += scale_score * self.quality_weights['avg_scale']

        # Density uniformity (lower variance is better, but not zero)
        density_var = ply.get('density_variance', 0)
        if 10 <= density_var <= 100:
            density_score = 100
        elif density_var < 10:
            density_score = 50  # Too uniform might mean sparse
        else:
            density_score = max(0, 100 - (density_var - 100) / 2)
        score += density_score * self.quality_weights['density_uniformity']

        return min(max(score, 0), 100)  # Clamp to 0-100

    def load_submissions(self) -> List[Dict]:
        """Load all submissions from database"""
        if not self.db_path.exists():
            return []

        submissions = []
        with open(self.db_path, 'r') as f:
            for line in f:
                try:
                    submissions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return submissions

    def analyze_quality_distribution(self):
        """Analyze your generation quality distribution"""
        submissions = self.load_submissions()

        if not submissions:
            print("❌ No submissions found. Generate some first!")
            return

        print("=" * 80)
        print("SELF-EVALUATION REPORT")
        print("=" * 80)
        print(f"\nTotal generations analyzed: {len(submissions)}")

        # Calculate quality scores for all submissions
        scored = []
        for sub in submissions:
            score = self.calculate_quality_score(sub)
            scored.append({
                'submission': sub,
                'quality_score': score
            })

        scores = [s['quality_score'] for s in scored]

        print(f"\n📊 Quality Score Distribution:")
        print(f"   Average: {statistics.mean(scores):.1f}/100")
        print(f"   Median:  {statistics.median(scores):.1f}/100")
        print(f"   Best:    {max(scores):.1f}/100")
        print(f"   Worst:   {min(scores):.1f}/100")
        print(f"   Stdev:   {statistics.stdev(scores) if len(scores) > 1 else 0:.1f}")

        # Quality tiers
        excellent = sum(1 for s in scores if s >= 80)
        good = sum(1 for s in scores if 60 <= s < 80)
        poor = sum(1 for s in scores if 40 <= s < 60)
        bad = sum(1 for s in scores if s < 40)

        print(f"\n📈 Quality Breakdown:")
        print(f"   Excellent (80-100): {excellent} ({excellent/len(scores)*100:.1f}%)")
        print(f"   Good (60-79):       {good} ({good/len(scores)*100:.1f}%)")
        print(f"   Poor (40-59):       {poor} ({poor/len(scores)*100:.1f}%)")
        print(f"   Bad (0-39):         {bad} ({bad/len(scores)*100:.1f}%)")

        # Identify best and worst
        best = max(scored, key=lambda x: x['quality_score'])
        worst = min(scored, key=lambda x: x['quality_score'])

        print("\n" + "=" * 80)
        print("🏆 BEST GENERATION")
        print("=" * 80)
        self._print_generation_details(best['submission'], best['quality_score'])

        print("\n" + "=" * 80)
        print("❌ WORST GENERATION")
        print("=" * 80)
        self._print_generation_details(worst['submission'], worst['quality_score'])

        # Patterns analysis
        print("\n" + "=" * 80)
        print("🔍 PATTERN ANALYSIS")
        print("=" * 80)

        top_20_pct = sorted(scored, key=lambda x: x['quality_score'], reverse=True)[:max(1, len(scored)//5)]
        bottom_20_pct = sorted(scored, key=lambda x: x['quality_score'])[:max(1, len(scored)//5)]

        # Compare metrics
        for metric in ['gaussian_count', 'file_size_mb', 'generation_time']:
            top_avg = statistics.mean([s['submission'][metric] for s in top_20_pct if metric in s['submission']])
            bottom_avg = statistics.mean([s['submission'][metric] for s in bottom_20_pct if metric in s['submission']])
            diff_pct = ((top_avg - bottom_avg) / bottom_avg * 100) if bottom_avg > 0 else 0

            print(f"\n{metric}:")
            print(f"   Top 20%:    {top_avg:,.2f}")
            print(f"   Bottom 20%: {bottom_avg:,.2f}")
            print(f"   Difference: {diff_pct:+.1f}%")

        # PLY metrics comparison
        for metric in ['spatial_variance', 'bbox_volume', 'avg_opacity', 'avg_scale']:
            top_values = [s['submission']['ply_quality_metrics'].get(metric, 0)
                         for s in top_20_pct if s['submission'].get('ply_quality_metrics')]
            bottom_values = [s['submission']['ply_quality_metrics'].get(metric, 0)
                            for s in bottom_20_pct if s['submission'].get('ply_quality_metrics')]

            if top_values and bottom_values:
                top_avg = statistics.mean(top_values)
                bottom_avg = statistics.mean(bottom_values)
                diff_pct = ((top_avg - bottom_avg) / bottom_avg * 100) if bottom_avg > 0 else 0

                print(f"\n{metric}:")
                print(f"   Top 20%:    {top_avg:.6f}")
                print(f"   Bottom 20%: {bottom_avg:.6f}")
                print(f"   Difference: {diff_pct:+.1f}%")

                if abs(diff_pct) > 50:
                    print(f"   🔴 MAJOR FACTOR - Focus on improving this!")
                elif abs(diff_pct) > 20:
                    print(f"   🟡 SIGNIFICANT FACTOR")

        print("\n" + "=" * 80)

    def _print_generation_details(self, submission: Dict, quality_score: float):
        """Print detailed info about a generation"""
        print(f"\nQuality Score: {quality_score:.1f}/100")
        print(f"Prompt: {submission.get('prompt', 'N/A')[:80]}")
        print(f"Gaussian Count: {submission.get('gaussian_count', 0):,}")
        print(f"File Size: {submission.get('file_size_mb', 0):.1f} MB")
        print(f"Generation Time: {submission.get('generation_time', 0):.1f}s")

        if submission.get('ply_quality_metrics'):
            ply = submission['ply_quality_metrics']
            print(f"\nPLY Quality Metrics:")
            print(f"   Spatial Variance:  {ply.get('spatial_variance', 0):.4f}")
            print(f"   Bbox Volume:       {ply.get('bbox_volume', 0):.4f}")
            print(f"   Avg Opacity:       {ply.get('avg_opacity', 0):.3f}")
            print(f"   Avg Scale:         {ply.get('avg_scale', 0):.4f}")
            print(f"   Density Variance:  {ply.get('density_variance', 0):.2f}")


if __name__ == "__main__":
    evaluator = SelfEvaluator()
    evaluator.analyze_quality_distribution()
