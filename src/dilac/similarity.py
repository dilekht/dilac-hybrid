"""
Lesk-ar: Arabic Semantic Similarity Measure
============================================

Implementation of the Lesk algorithm adapted for Arabic using DiLAC.

Based on the original Lesk algorithm and adaptations described in:
- Lesk (1986): Automatic sense disambiguation using machine readable dictionaries
- Banerjee & Pedersen (2002): An adapted Lesk algorithm for word sense disambiguation
"""

import math
import re
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArabicPreprocessor:
    """Preprocessor for Arabic text normalization and tokenization"""
    
    # Arabic stopwords (common function words)
    STOPWORDS = {
        'من', 'إلى', 'على', 'في', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
        'التي', 'الذي', 'الذين', 'اللذين', 'اللتين', 'اللواتي', 'هو', 'هي',
        'هم', 'هن', 'نحن', 'أنا', 'أنت', 'أنتم', 'أنتن', 'أنتما',
        'كان', 'كانت', 'كانوا', 'يكون', 'تكون', 'ليس', 'ليست',
        'قد', 'لقد', 'إن', 'أن', 'لأن', 'كأن', 'لكن', 'ثم', 'أو', 'و', 'ف',
        'لا', 'لم', 'لن', 'ما', 'من', 'كل', 'بعض', 'غير', 'مثل', 'بين',
        'عند', 'قبل', 'بعد', 'فوق', 'تحت', 'حول', 'ضد', 'منذ', 'خلال',
        'حتى', 'لدى', 'نحو', 'إلا', 'سوى', 'دون', 'وراء', 'أمام',
        'أي', 'أيضا', 'جدا', 'كثيرا', 'قليلا', 'فقط', 'كذلك', 'أيها',
        'يا', 'آه', 'إذ', 'إذا', 'لو', 'مما', 'إما', 'أما', 'بل',
    }
    
    # Diacritics pattern
    DIACRITICS = re.compile(r'[\u064B-\u0652\u0670]')
    
    # Punctuation
    PUNCTUATION = re.compile(r'[^\w\s\u0600-\u06FF]')
    
    @classmethod
    def normalize(cls, text: str) -> str:
        """Normalize Arabic text"""
        # Remove diacritics
        text = cls.DIACRITICS.sub('', text)
        
        # Normalize alef variants
        text = re.sub(r'[إأآ]', 'ا', text)
        
        # Normalize teh marbuta
        text = text.replace('ة', 'ه')
        
        # Normalize alef maksura
        text = text.replace('ى', 'ي')
        
        # Remove tatweel
        text = text.replace('ـ', '')
        
        return text
    
    @classmethod
    def tokenize(cls, text: str, remove_stopwords: bool = True) -> List[str]:
        """Tokenize and optionally remove stopwords"""
        # Normalize
        text = cls.normalize(text)
        
        # Remove punctuation
        text = cls.PUNCTUATION.sub(' ', text)
        
        # Split
        tokens = text.split()
        
        # Filter
        tokens = [t for t in tokens if len(t) > 1]
        
        if remove_stopwords:
            tokens = [t for t in tokens if t not in cls.STOPWORDS]
        
        return tokens


class LeskAr:
    """
    Lesk-ar: Arabic Semantic Similarity Measure
    
    Calculates similarity between Arabic words based on gloss overlap,
    adapted for the DiLAC dictionary structure.
    
    Features:
    - Uses definitions and examples from DiLAC
    - Supports IDF weighting
    - Integrates domain information
    - Handles Arabic-specific preprocessing
    """
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize Lesk-ar measure.
        
        Args:
            database_path: Path to DiLAC Lesk format JSON file
        """
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.word_frequencies: Dict[str, int] = {}
        self.entries: Dict[str, Dict] = {}  # lemma -> entry data
        self.total_definitions = 0
        
        if database_path:
            self.load_database(database_path)
    
    def load_database(self, filepath: str):
        """Load DiLAC-Lesk format database"""
        logger.info(f"Loading database from {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.word_to_id = data.get('word_to_id', {})
        self.id_to_word = {int(k): v for k, v in data.get('id_to_word', {}).items()}
        self.word_frequencies = data.get('word_frequencies', {})
        
        # Index entries by lemma
        for entry in data.get('entries', []):
            lemma = entry['lemma']
            if lemma not in self.entries:
                self.entries[lemma] = entry
            else:
                # Merge senses for same lemma
                self.entries[lemma]['senses'].extend(entry['senses'])
        
        # Count total definitions
        self.total_definitions = sum(
            len(entry['senses']) for entry in self.entries.values()
        )
        
        logger.info(f"Loaded {len(self.entries)} entries, {self.total_definitions} senses")
    
    def get_gloss_words(self, lemma: str) -> Set[int]:
        """Get all gloss word IDs for a lemma"""
        entry = self.entries.get(lemma)
        if not entry:
            return set()
        
        words = set()
        for sense in entry['senses']:
            words.update(sense.get('encoded_gloss', []))
        
        return words
    
    def get_sense_gloss_words(self, lemma: str, sense_index: int = 0) -> Set[int]:
        """Get gloss word IDs for a specific sense"""
        entry = self.entries.get(lemma)
        if not entry or sense_index >= len(entry['senses']):
            return set()
        
        return set(entry['senses'][sense_index].get('encoded_gloss', []))
    
    def get_domains(self, lemma: str) -> List[str]:
        """Get all domains for a lemma's senses"""
        entry = self.entries.get(lemma)
        if not entry:
            return []
        
        domains = []
        for sense in entry['senses']:
            if sense.get('domain'):
                domains.append(sense['domain'])
        
        return domains
    
    def overlap(self, words1: Set[int], words2: Set[int]) -> int:
        """Calculate basic overlap between two sets of words"""
        return len(words1 & words2)
    
    def weighted_overlap(self, words1: Set[int], words2: Set[int]) -> float:
        """Calculate IDF-weighted overlap"""
        common = words1 & words2
        if not common:
            return 0.0
        
        total_weight = 0.0
        for word_id in common:
            word = self.id_to_word.get(word_id, "")
            if word:
                # Calculate IDF weight
                freq = self.word_frequencies.get(word, 1)
                idf = math.log(self.total_definitions / freq) if freq > 0 else 0
                total_weight += idf
        
        return total_weight
    
    def similarity(
        self,
        word1: str,
        word2: str,
        use_weighting: bool = True,
        use_domain: bool = True,
        normalize: bool = True
    ) -> float:
        """
        Calculate semantic similarity between two Arabic words.
        
        Args:
            word1: First Arabic word
            word2: Second Arabic word
            use_weighting: Use IDF weighting
            use_domain: Boost score if words share domain
            normalize: Normalize by gloss size
        
        Returns:
            Similarity score in [0, 1]
        """
        # Normalize input
        word1 = ArabicPreprocessor.normalize(word1)
        word2 = ArabicPreprocessor.normalize(word2)
        
        # Get gloss words
        gloss1 = self.get_gloss_words(word1)
        gloss2 = self.get_gloss_words(word2)
        
        if not gloss1 or not gloss2:
            return 0.0
        
        # Calculate overlap
        if use_weighting:
            overlap_score = self.weighted_overlap(gloss1, gloss2)
        else:
            overlap_score = float(self.overlap(gloss1, gloss2))
        
        # Normalize by gloss size (logarithmic)
        if normalize:
            max_size = max(len(gloss1), len(gloss2))
            if max_size > 0:
                # Use log normalization as in Eq. 6.5
                overlap_score = overlap_score / math.log2(max_size + 1)
        
        # Domain boost
        if use_domain:
            domains1 = set(self.get_domains(word1))
            domains2 = set(self.get_domains(word2))
            
            if domains1 & domains2:
                overlap_score *= 1.2  # 20% boost for shared domain
        
        # Scale to [0, 1]
        max_possible = self._estimate_max_similarity()
        if max_possible > 0:
            overlap_score = min(1.0, overlap_score / max_possible)
        
        return overlap_score
    
    def _estimate_max_similarity(self) -> float:
        """Estimate maximum possible similarity for scaling"""
        # Use average IDF as reference
        if not self.word_frequencies:
            return 1.0
        
        avg_idf = sum(
            math.log(self.total_definitions / max(f, 1))
            for f in self.word_frequencies.values()
        ) / len(self.word_frequencies)
        
        return avg_idf * 10  # Approximate max overlap
    
    def best_sense_pair(
        self,
        word1: str,
        word2: str
    ) -> Tuple[int, int, float]:
        """
        Find the best matching sense pair between two words.
        
        Returns:
            Tuple of (sense_index1, sense_index2, similarity_score)
        """
        word1 = ArabicPreprocessor.normalize(word1)
        word2 = ArabicPreprocessor.normalize(word2)
        
        entry1 = self.entries.get(word1)
        entry2 = self.entries.get(word2)
        
        if not entry1 or not entry2:
            return (0, 0, 0.0)
        
        best_score = 0.0
        best_pair = (0, 0)
        
        for i, sense1 in enumerate(entry1['senses']):
            gloss1 = set(sense1.get('encoded_gloss', []))
            
            for j, sense2 in enumerate(entry2['senses']):
                gloss2 = set(sense2.get('encoded_gloss', []))
                
                score = self.weighted_overlap(gloss1, gloss2)
                
                # Domain matching bonus
                if sense1.get('domain') and sense1.get('domain') == sense2.get('domain'):
                    score *= 1.3
                
                if score > best_score:
                    best_score = score
                    best_pair = (i, j)
        
        return (*best_pair, best_score)


class ExtendedLesk(LeskAr):
    """
    Extended Lesk algorithm that also considers:
    - Hypernyms and hyponyms
    - Related terms
    - Multi-word expressions
    """
    
    def __init__(self, database_path: Optional[str] = None):
        super().__init__(database_path)
        self.relations: Dict[str, Dict[str, List[str]]] = {}
    
    def add_relations(self, relations_path: str):
        """Load semantic relations"""
        with open(relations_path, 'r', encoding='utf-8') as f:
            self.relations = json.load(f)
    
    def get_extended_gloss(self, lemma: str, depth: int = 1) -> Set[int]:
        """Get gloss words including related words"""
        words = self.get_gloss_words(lemma)
        
        if depth > 0 and lemma in self.relations:
            # Add hypernym glosses
            for hypernym in self.relations[lemma].get('hypernyms', []):
                words.update(self.get_extended_gloss(hypernym, depth - 1))
            
            # Add hyponym glosses
            for hyponym in self.relations[lemma].get('hyponyms', []):
                words.update(self.get_extended_gloss(hyponym, depth - 1))
        
        return words
    
    def extended_similarity(
        self,
        word1: str,
        word2: str,
        depth: int = 1
    ) -> float:
        """Calculate similarity using extended glosses"""
        word1 = ArabicPreprocessor.normalize(word1)
        word2 = ArabicPreprocessor.normalize(word2)
        
        gloss1 = self.get_extended_gloss(word1, depth)
        gloss2 = self.get_extended_gloss(word2, depth)
        
        if not gloss1 or not gloss2:
            return 0.0
        
        overlap_score = self.weighted_overlap(gloss1, gloss2)
        
        # Normalize
        max_size = max(len(gloss1), len(gloss2))
        if max_size > 0:
            overlap_score = overlap_score / math.log2(max_size + 1)
        
        # Scale
        max_possible = self._estimate_max_similarity()
        if max_possible > 0:
            overlap_score = min(1.0, overlap_score / max_possible)
        
        return overlap_score


class SimilarityBenchmark:
    """
    Benchmark for evaluating similarity measures.
    
    Uses AWSS (Arabic Word Semantic Similarity) benchmark dataset.
    """
    
    def __init__(self, similarity_measure: LeskAr):
        self.measure = similarity_measure
        self.results: List[Dict] = []
    
    def load_benchmark(self, filepath: str) -> List[Tuple[str, str, float]]:
        """Load benchmark dataset"""
        pairs = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data['pairs']:
            pairs.append((
                item['word1'],
                item['word2'],
                item['human_score']
            ))
        
        return pairs
    
    def evaluate(
        self,
        benchmark_pairs: List[Tuple[str, str, float]]
    ) -> Dict:
        """
        Evaluate similarity measure on benchmark.
        
        Returns:
            Dictionary with MSE, correlation, and detailed results
        """
        self.results = []
        
        predicted_scores = []
        human_scores = []
        
        for word1, word2, human_score in benchmark_pairs:
            pred_score = self.measure.similarity(word1, word2)
            
            self.results.append({
                'word1': word1,
                'word2': word2,
                'human_score': human_score,
                'predicted_score': pred_score,
                'error': pred_score - human_score,
                'squared_error': (pred_score - human_score) ** 2
            })
            
            predicted_scores.append(pred_score)
            human_scores.append(human_score)
        
        # Calculate metrics
        mse = sum(r['squared_error'] for r in self.results) / len(self.results)
        correlation = self._pearson_correlation(predicted_scores, human_scores)
        
        # Calculate by similarity category
        low_sim = [r for r in self.results if r['human_score'] < 0.33]
        med_sim = [r for r in self.results if 0.33 <= r['human_score'] < 0.66]
        high_sim = [r for r in self.results if r['human_score'] >= 0.66]
        
        return {
            'mse': mse,
            'correlation': correlation,
            'mse_low': sum(r['squared_error'] for r in low_sim) / max(len(low_sim), 1),
            'mse_medium': sum(r['squared_error'] for r in med_sim) / max(len(med_sim), 1),
            'mse_high': sum(r['squared_error'] for r in high_sim) / max(len(high_sim), 1),
            'correlation_low': self._pearson_correlation(
                [r['predicted_score'] for r in low_sim],
                [r['human_score'] for r in low_sim]
            ),
            'correlation_medium': self._pearson_correlation(
                [r['predicted_score'] for r in med_sim],
                [r['human_score'] for r in med_sim]
            ),
            'correlation_high': self._pearson_correlation(
                [r['predicted_score'] for r in high_sim],
                [r['human_score'] for r in high_sim]
            ),
            'total_pairs': len(self.results),
            'details': self.results
        }
    
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


if __name__ == "__main__":
    # Example usage
    lesk = LeskAr("data/processed/dilac_lesk.json")
    
    # Test similarity
    pairs = [
        ("ساحل", "شاطئ"),  # coast - shore
        ("كتاب", "مجلد"),  # book - volume
        ("سيارة", "حافلة"), # car - bus
    ]
    
    for w1, w2 in pairs:
        score = lesk.similarity(w1, w2)
        print(f"Similarity({w1}, {w2}) = {score:.4f}")
