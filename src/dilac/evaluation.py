"""
DiLAC Evaluation Framework
===========================

Comprehensive evaluation module for:
1. Semantic similarity measures (Lesk-ar vs Wup vs AWSS)
2. WSD accuracy
3. Comparison with benchmarks from Table 6.2
"""

import json
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# AWSS Benchmark Dataset - 40 pairs from Table 6.2
# Scaled to [0, 1] by dividing by 4
AWSS_BENCHMARK_40 = [
    # Low similarity (12 pairs)
    ("Coast", "Endorsement", "ساحل", "تصديق", 0.01),
    ("Noon", "String", "ظهر", "خيط", 0.01),
    ("Stove", "Walk", "موقد", "مشي", 0.02),
    ("Cord", "Midday", "حبل", "ظهيرة", 0.02),
    ("Signature", "String", "توقيع", "خيط", 0.02),
    ("Boy", "Endorsement", "صبي", "تصديق", 0.03),
    ("Boy", "Midday", "صبي", "ظهيرة", 0.04),
    ("Smile", "Village", "ابتسامة", "قرية", 0.05),
    ("Noon", "Fasting", "ظهر", "صيام", 0.07),
    ("Glass", "Diamond", "كأس", "الماس", 0.09),
    ("Sepulcher", "Sheikh", "ضريح", "شيخ", 0.22),
    ("Countryside", "Vegetable", "ريف", "خضار", 0.31),
    
    # Medium similarity (13 pairs)
    ("Tumbler", "Tool", "قدح", "أداة", 0.33),
    ("Laugh", "Feast", "ضحك", "عيد", 0.34),
    ("Girl", "Odalisque", "فتاة", "جارية", 0.49),
    ("Feast", "Fasting", "عيد", "صيام", 0.49),
    ("Coach", "Means", "حافلة", "وسيلة", 0.52),
    ("Sage", "Sheikh", "حكيم", "شيخ", 0.57),
    ("Girl", "Sister", "فتاة", "اخت", 0.60),
    ("Hen", "Pigeon", "دجاجة", "حمامة", 0.65),
    ("Hill", "Mountain", "تل", "جبل", 0.65),
    ("Master", "Sheikh", "سيد", "شيخ", 0.67),
    ("Food", "Vegetable", "طعام", "خضار", 0.69),
    ("Slave", "Odalisque", "عبد", "جارية", 0.71),
    ("Run", "Walk", "جري", "مشي", 0.75),
    
    # High similarity (15 pairs)
    ("Cord", "String", "حبل", "خيط", 0.77),
    ("Forest", "Woodland", "غابة", "أحراش", 0.79),
    ("Sage", "Thinker", "حكيم", "مفكر", 0.83),
    ("Journey", "Travel", "رحلة", "سفر", 0.85),
    ("Gem", "Diamond", "جوهرة", "الماس", 0.85),
    ("Countryside", "Village", "ريف", "قرية", 0.85),
    ("Cushion", "Pillow", "مسند", "مخدة", 0.85),
    ("Smile", "Laugh", "ابتسامة", "ضحك", 0.87),
    ("Signature", "Endorsement", "توقيع", "تصديق", 0.90),
    ("Tool", "Means", "أداة", "وسيلة", 0.92),
    ("Sepulcher", "Grave", "ضريح", "قبر", 0.94),
    ("Boy", "Lad", "صبي", "فتى", 0.93),
    ("Wizard", "Magician", "ساحر", "مشعوذ", 0.94),
    ("Coach", "Bus", "حافلة", "باص", 0.95),
    ("Glass", "Tumbler", "كأس", "قدح", 0.95),
]


# Reference results from Table 6.2
REFERENCE_RESULTS = {
    "Wup_AWN": {
        "mse": 0.016540541,
        "correlation": 0.941168501,
        "mse_low": 0.002754545,
        "mse_medium": 0.028141667,
        "mse_high": 0.017428571,
        "correlation_low": 0.918748881,
        "correlation_medium": 0.598689836,
        "correlation_high": 0.41198363
    },
    "AWSS": {
        "mse": 0.026978378,
        "correlation": 0.889771136,
        "mse_low": 0.007563636,
        "mse_medium": 0.045258333,
        "mse_high": 0.026564286,
        "correlation_low": 0.756095628,
        "correlation_medium": 0.293223608,
        "correlation_high": 0.350032563
    },
    "Lesk_ar_DiLAC": {
        "mse": 0.020308627,
        "correlation": 0.916607931,
        "mse_low": 0.008284841,
        "mse_medium": 0.023911111,
        "mse_high": 0.026805504,
        "correlation_low": 0.840386909,
        "correlation_medium": 0.657098832,
        "correlation_high": 0.31006654
    }
}


@dataclass
class EvaluationResult:
    """Result of a single pair evaluation"""
    word1_ar: str
    word2_ar: str
    word1_en: str
    word2_en: str
    human_score: float
    predicted_score: float
    error: float
    squared_error: float
    similarity_category: str


@dataclass
class EvaluationMetrics:
    """Complete evaluation metrics"""
    mse: float
    correlation: float
    mse_low: float
    mse_medium: float
    mse_high: float
    correlation_low: float
    correlation_medium: float
    correlation_high: float
    total_pairs: int
    details: List[EvaluationResult] = field(default_factory=list)


class SimilarityEvaluator:
    """Evaluator for semantic similarity measures"""
    
    def __init__(self, similarity_measure):
        self.measure = similarity_measure
    
    def evaluate_benchmark(self, benchmark: List[Tuple] = None) -> EvaluationMetrics:
        """Evaluate on AWSS benchmark"""
        if benchmark is None:
            benchmark = AWSS_BENCHMARK_40
        
        results = []
        
        for item in benchmark:
            word1_en, word2_en, word1_ar, word2_ar, human_score = item
            
            try:
                predicted = self.measure.similarity(word1_ar, word2_ar)
            except Exception as e:
                logger.warning(f"Error: {e}")
                predicted = 0.0
            
            error = predicted - human_score
            sq_error = error ** 2
            
            if human_score < 0.33:
                category = 'low'
            elif human_score < 0.66:
                category = 'medium'
            else:
                category = 'high'
            
            results.append(EvaluationResult(
                word1_ar=word1_ar,
                word2_ar=word2_ar,
                word1_en=word1_en,
                word2_en=word2_en,
                human_score=human_score,
                predicted_score=predicted,
                error=error,
                squared_error=sq_error,
                similarity_category=category
            ))
        
        return self._compute_metrics(results)
    
    def _compute_metrics(self, results: List[EvaluationResult]) -> EvaluationMetrics:
        """Compute aggregate metrics"""
        mse = sum(r.squared_error for r in results) / len(results)
        
        predicted = [r.predicted_score for r in results]
        human = [r.human_score for r in results]
        correlation = self._pearson_correlation(predicted, human)
        
        low = [r for r in results if r.similarity_category == 'low']
        medium = [r for r in results if r.similarity_category == 'medium']
        high = [r for r in results if r.similarity_category == 'high']
        
        mse_low = sum(r.squared_error for r in low) / max(len(low), 1)
        mse_medium = sum(r.squared_error for r in medium) / max(len(medium), 1)
        mse_high = sum(r.squared_error for r in high) / max(len(high), 1)
        
        corr_low = self._pearson_correlation(
            [r.predicted_score for r in low],
            [r.human_score for r in low]
        ) if low else 0.0
        
        corr_medium = self._pearson_correlation(
            [r.predicted_score for r in medium],
            [r.human_score for r in medium]
        ) if medium else 0.0
        
        corr_high = self._pearson_correlation(
            [r.predicted_score for r in high],
            [r.human_score for r in high]
        ) if high else 0.0
        
        return EvaluationMetrics(
            mse=mse,
            correlation=correlation,
            mse_low=mse_low,
            mse_medium=mse_medium,
            mse_high=mse_high,
            correlation_low=corr_low,
            correlation_medium=corr_medium,
            correlation_high=corr_high,
            total_pairs=len(results),
            details=results
        )
    
    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) < 2 or len(y) < 2 or len(x) != len(y):
            return 0.0
        
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)
        
        denominator = math.sqrt(var_x * var_y)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def compare_with_references(self, metrics: EvaluationMetrics) -> Dict:
        """Compare with Table 6.2 reference results"""
        comparison = {}
        
        for ref_name, ref_metrics in REFERENCE_RESULTS.items():
            comparison[ref_name] = {
                'mse_diff': metrics.mse - ref_metrics['mse'],
                'correlation_diff': metrics.correlation - ref_metrics['correlation'],
                'mse_better': metrics.mse < ref_metrics['mse'],
                'correlation_better': metrics.correlation > ref_metrics['correlation']
            }
        
        return comparison
    
    def generate_markdown_report(self, metrics: EvaluationMetrics) -> str:
        """Generate Markdown evaluation report"""
        lines = [
            "# DiLAC Similarity Evaluation Report",
            "",
            f"**Total pairs evaluated:** {metrics.total_pairs}",
            "",
            "## Overall Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| MSE | {metrics.mse:.6f} |",
            f"| Correlation | {metrics.correlation:.6f} |",
            "",
            "## Results by Similarity Category",
            "",
            "| Category | MSE | Correlation |",
            "|----------|-----|-------------|",
            f"| Low | {metrics.mse_low:.6f} | {metrics.correlation_low:.6f} |",
            f"| Medium | {metrics.mse_medium:.6f} | {metrics.correlation_medium:.6f} |",
            f"| High | {metrics.mse_high:.6f} | {metrics.correlation_high:.6f} |",
            "",
            "## Comparison with Reference Methods (Table 6.2)",
            "",
            "| Method | MSE | Correlation |",
            "|--------|-----|-------------|",
        ]
        
        for ref_name, ref_metrics in REFERENCE_RESULTS.items():
            lines.append(f"| {ref_name} | {ref_metrics['mse']:.6f} | {ref_metrics['correlation']:.6f} |")
        
        return "\n".join(lines)


def save_benchmark_json(output_path: str):
    """Save benchmark dataset to JSON"""
    data = {
        'name': 'AWSS Arabic Word Semantic Similarity Benchmark',
        'description': 'Human similarity judgments for Arabic word pairs',
        'source': 'Fazza et al. (2012)',
        'pairs': [
            {
                'word1_en': item[0],
                'word2_en': item[1],
                'word1': item[2],
                'word2': item[3],
                'human_score': item[4]
            }
            for item in AWSS_BENCHMARK_40
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved benchmark to {output_path}")


if __name__ == "__main__":
    save_benchmark_json("data/benchmarks/awss_40.json")
