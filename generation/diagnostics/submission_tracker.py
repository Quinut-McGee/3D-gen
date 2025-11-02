"""
Submission Correlation Tracker
Logs every submission with detailed metrics for pattern analysis
"""
import json
import time
from pathlib import Path
from typing import Dict, Optional
from loguru import logger


class SubmissionTracker:
    """Track all submissions and their validator feedback for correlation analysis"""

    def __init__(self, db_path: str = "/tmp/submission_database.jsonl"):
        self.db_path = Path(db_path)
        logger.info(f"Submission tracker initialized: {self.db_path}")

    def log_submission(
        self,
        prompt: str,
        gaussian_count: int,
        file_size_mb: float,
        generation_time: float,
        ply_quality_metrics: Dict[str, float],
        clip_score_2d: Optional[float] = None,
        clip_score_3d: Optional[float] = None,
    ) -> str:
        """
        Log a submission BEFORE it's sent to validator.
        Returns submission_id for later correlation with validator feedback.
        """
        submission_id = f"{int(time.time())}_{hash(prompt) % 100000}"

        record = {
            'submission_id': submission_id,
            'timestamp': time.time(),
            'prompt': prompt,
            'gaussian_count': gaussian_count,
            'file_size_mb': file_size_mb,
            'generation_time': generation_time,
            'ply_quality_metrics': ply_quality_metrics,
            'clip_score_2d': clip_score_2d,
            'clip_score_3d': clip_score_3d,
            'validator_score': None,  # Will be filled by update_validator_feedback()
            'validator_uid': None,
        }

        # Append to JSONL database
        try:
            with open(self.db_path, 'a') as f:
                f.write(json.dumps(record) + '\n')
            logger.debug(f"Logged submission {submission_id}")
        except Exception as e:
            logger.error(f"Failed to log submission: {e}")

        return submission_id

    def update_validator_feedback(
        self,
        submission_id: str,
        validator_score: float,
        validator_uid: int
    ):
        """
        Update a submission record with validator feedback.
        Called when validator response is received.
        """
        try:
            # Read all records
            if not self.db_path.exists():
                logger.warning(f"Database not found: {self.db_path}")
                return

            records = []
            with open(self.db_path, 'r') as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

            # Find and update the record
            updated = False
            for record in records:
                if record.get('submission_id') == submission_id:
                    record['validator_score'] = validator_score
                    record['validator_uid'] = validator_uid
                    updated = True
                    logger.info(f"Updated {submission_id} with Score={validator_score} from validator {validator_uid}")
                    break

            if updated:
                # Rewrite entire database (not efficient, but simple for MVP)
                with open(self.db_path, 'w') as f:
                    for record in records:
                        f.write(json.dumps(record) + '\n')
            else:
                logger.warning(f"Submission {submission_id} not found in database")

        except Exception as e:
            logger.error(f"Failed to update validator feedback: {e}")

    def get_statistics(self) -> Dict:
        """
        Analyze all submissions and return statistics.
        """
        try:
            if not self.db_path.exists():
                return {'error': 'No data yet'}

            records = []
            with open(self.db_path, 'r') as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

            if not records:
                return {'error': 'No valid records'}

            # Separate score=0.0 from score>0
            score_zero = [r for r in records if r.get('validator_score') == 0.0]
            score_good = [r for r in records if r.get('validator_score') is not None and r.get('validator_score') > 0.0]
            score_pending = [r for r in records if r.get('validator_score') is None]

            stats = {
                'total_submissions': len(records),
                'score_zero_count': len(score_zero),
                'score_good_count': len(score_good),
                'pending_count': len(score_pending),
                'score_zero_rate': len(score_zero) / len(records) if records else 0,
            }

            # Compare metrics between score=0 and score>0
            if score_zero and score_good:
                stats['avg_gaussians_score_zero'] = sum(r['gaussian_count'] for r in score_zero) / len(score_zero)
                stats['avg_gaussians_score_good'] = sum(r['gaussian_count'] for r in score_good) / len(score_good)

                stats['avg_filesize_score_zero'] = sum(r['file_size_mb'] for r in score_zero) / len(score_zero)
                stats['avg_filesize_score_good'] = sum(r['file_size_mb'] for r in score_good) / len(score_good)

                # PLY quality metrics comparison
                if score_zero[0].get('ply_quality_metrics'):
                    for metric_name in ['spatial_variance', 'bbox_volume', 'avg_opacity', 'avg_scale']:
                        zero_values = [r['ply_quality_metrics'].get(metric_name, 0) for r in score_zero if r.get('ply_quality_metrics')]
                        good_values = [r['ply_quality_metrics'].get(metric_name, 0) for r in score_good if r.get('ply_quality_metrics')]

                        if zero_values and good_values:
                            stats[f'{metric_name}_score_zero'] = sum(zero_values) / len(zero_values)
                            stats[f'{metric_name}_score_good'] = sum(good_values) / len(good_values)

            return stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {'error': str(e)}


# Global instance
_tracker = None


def get_tracker() -> SubmissionTracker:
    """Get or create global tracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = SubmissionTracker()
    return _tracker
