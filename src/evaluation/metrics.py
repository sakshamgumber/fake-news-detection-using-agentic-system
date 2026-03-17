"""
Evaluation Metrics - Comprehensive metrics for fact-checking performance
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import statistics
from collections import defaultdict
from loguru import logger

@dataclass
class ClassificationMetrics:
    """Standard classification metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    mean_processing_time: float
    total_processing_time: float
    mean_queries_per_claim: float
    mean_evidence_per_claim: float
    mean_high_credibility_ratio: float


@dataclass
class ExplanationMetrics:
    """Explanation quality metrics"""
    mean_coverage: float
    mean_soundness: float
    mean_readability: float
    mean_overall_quality: float


@dataclass
class ComprehensiveMetrics:
    """All metrics combined"""
    classification: ClassificationMetrics
    performance: PerformanceMetrics
    explanation: ExplanationMetrics
    by_category: Dict[str, ClassificationMetrics]
    by_difficulty: Dict[str, ClassificationMetrics]


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics"""

    @staticmethod
    def calculate_classification_metrics(
        predictions: List[str],
        ground_truths: List[str]
    ) -> ClassificationMetrics:
        """
        Calculate classification metrics (accuracy, precision, recall, F1).

        Args:
            predictions: List of predicted verdicts
            ground_truths: List of ground truth labxels

        Returns:
            ClassificationMetrics object
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")

        tp = tn = fp = fn = 0
        correct = 0

        for pred, truth in zip(predictions, ground_truths):
            if pred == truth:
                correct += 1
                
            p_is_pos = (pred == "SUPPORTED")
            t_is_pos = (truth == "SUPPORTED")

            if p_is_pos and t_is_pos:
                tp += 1
            elif not p_is_pos and not t_is_pos:
                tn += 1
            elif p_is_pos and not t_is_pos:
                fp += 1
            elif not p_is_pos and t_is_pos:
                fn += 1

        total = len(predictions)
        accuracy = correct / total if total > 0 else 0

        logger.info(f"tp is {tp} tn is {tn} fp is {fp} fn is {fn}")
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        logger.info(f"accurcy is {accuracy} precision is {precision} recall is {recall} f1_score is {f1_score}")
        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn
        )

