"""
Arabic Word Sense Disambiguation (WSD) Algorithm
==================================================

Implementation of WSD algorithms for Arabic using DiLAC:
1. Simplified Lesk Algorithm (sense-based)
2. Context-based WSD
3. Domain-aware WSD

Based on the methodology described in Chapter 6.
"""

import math
import re
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import json
import logging
from dataclasses import dataclass

from .similarity import LeskAr, ArabicPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DisambiguatedWord:
    """Result of word sense disambiguation"""
    word: str
    lemma: str
    selected_sense_id: str
    selected_sense_definition: str
    confidence_score: float
    all_sense_scores: List[Tuple[str, float]]
    context_window: List[str]


@dataclass
class Sense:
    """Sense representation for WSD"""
    id: str
    definition: str
    domain: Optional[str]
    encoded_gloss: Set[int]
    frequency_rank: int  # Lower is more frequent
    
    def __hash__(self):
        return hash(self.id)


class SimplifiedLesk:
    """
    Simplified Lesk Algorithm for Arabic WSD
    
    As described in Eq. 6.4 and 6.5:
    - overlap = |D(Sj) ∩ C(w)|
    - normalized_overlap = overlap / log2(|D(Sj)|)
    
    Where:
    - D(Sj) is the definition (gloss) of sense j
    - C(w) is the context window
    """
    
    def __init__(self, database: LeskAr):
        self.db = database
        self.preprocessor = ArabicPreprocessor()
    
    def disambiguate(
        self,
        target_word: str,
        context: str,
        window_size: int = 5
    ) -> Optional[DisambiguatedWord]:
        """
        Disambiguate a target word given its context.
        
        Args:
            target_word: The word to disambiguate
            context: The sentence or paragraph containing the word
            window_size: Number of context words on each side
        
        Returns:
            DisambiguatedWord with the best sense
        """
        # Normalize
        target_word = ArabicPreprocessor.normalize(target_word)
        
        # Get entry
        entry = self.db.entries.get(target_word)
        if not entry:
            logger.warning(f"Word '{target_word}' not found in dictionary")
            return None
        
        senses = entry.get('senses', [])
        if not senses:
            logger.warning(f"No senses found for '{target_word}'")
            return None
        
        # Extract context window
        context_words = self._extract_context_window(target_word, context, window_size)
        context_ids = self._encode_context(context_words)
        
        # Score each sense
        sense_scores = []
        
        for i, sense in enumerate(senses):
            gloss_ids = set(sense.get('encoded_gloss', []))
            
            # Calculate overlap (Eq. 6.4)
            overlap = len(gloss_ids & context_ids)
            
            # Normalize (Eq. 6.5)
            if len(gloss_ids) > 0:
                normalized_score = overlap / math.log2(len(gloss_ids) + 1)
            else:
                normalized_score = 0.0
            
            # Frequency bias: prefer more common senses
            # First sense is most frequent (rank 0)
            frequency_bias = 1.0 / (i + 1)
            
            # Combined score
            final_score = normalized_score + (0.1 * frequency_bias)
            
            sense_scores.append((sense['id'], final_score, sense))
        
        # Sort by score (descending)
        sense_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select best sense (or first if all scores are 0)
        best_sense_id, best_score, best_sense = sense_scores[0]
        
        # If no overlap, default to most frequent sense
        if best_score == 0:
            best_sense = senses[0]
            best_sense_id = best_sense['id']
        
        return DisambiguatedWord(
            word=target_word,
            lemma=entry.get('lemma', target_word),
            selected_sense_id=best_sense_id,
            selected_sense_definition=best_sense.get('definition', ''),
            confidence_score=best_score,
            all_sense_scores=[(s[0], s[1]) for s in sense_scores],
            context_window=context_words
        )
    
    def _extract_context_window(
        self,
        target: str,
        context: str,
        window_size: int
    ) -> List[str]:
        """Extract context window around target word"""
        # Tokenize
        tokens = ArabicPreprocessor.tokenize(context, remove_stopwords=False)
        
        # Find target position
        target_positions = []
        for i, token in enumerate(tokens):
            if ArabicPreprocessor.normalize(token) == target:
                target_positions.append(i)
        
        if not target_positions:
            # Target not found, use all tokens as context
            return ArabicPreprocessor.tokenize(context, remove_stopwords=True)
        
        # Extract window around first occurrence
        pos = target_positions[0]
        start = max(0, pos - window_size)
        end = min(len(tokens), pos + window_size + 1)
        
        window = tokens[start:pos] + tokens[pos+1:end]
        
        # Remove stopwords
        window = [w for w in window if w not in ArabicPreprocessor.STOPWORDS]
        
        return window
    
    def _encode_context(self, words: List[str]) -> Set[int]:
        """Convert context words to IDs"""
        ids = set()
        for word in words:
            normalized = ArabicPreprocessor.normalize(word)
            if normalized in self.db.word_to_id:
                ids.add(self.db.word_to_id[normalized])
        return ids


class ContextBasedWSD:
    """
    Context-based WSD using semantic similarity of neighbors.
    
    Uses the senses of neighboring words to find the best sense
    for the target word based on semantic overlap.
    """
    
    def __init__(self, database: LeskAr):
        self.db = database
        self.preprocessor = ArabicPreprocessor()
    
    def disambiguate(
        self,
        target_word: str,
        context: str,
        window_size: int = 5
    ) -> Optional[DisambiguatedWord]:
        """
        Disambiguate using context word senses.
        
        Args:
            target_word: The word to disambiguate
            context: The sentence containing the word
            window_size: Context window size
        
        Returns:
            DisambiguatedWord with the best sense
        """
        target_word = ArabicPreprocessor.normalize(target_word)
        
        entry = self.db.entries.get(target_word)
        if not entry:
            return None
        
        senses = entry.get('senses', [])
        if not senses:
            return None
        
        # Get context words
        context_tokens = ArabicPreprocessor.tokenize(context, remove_stopwords=True)
        context_words = [
            w for w in context_tokens 
            if ArabicPreprocessor.normalize(w) != target_word
        ][:window_size * 2]
        
        # Get glosses of context words (union of all senses)
        context_glosses = set()
        for cw in context_words:
            cw_norm = ArabicPreprocessor.normalize(cw)
            cw_entry = self.db.entries.get(cw_norm)
            if cw_entry:
                for sense in cw_entry.get('senses', []):
                    context_glosses.update(sense.get('encoded_gloss', []))
        
        # Score each sense of target word
        sense_scores = []
        
        for i, sense in enumerate(senses):
            gloss_ids = set(sense.get('encoded_gloss', []))
            
            # Calculate weighted overlap with context glosses
            common = gloss_ids & context_glosses
            
            if not common:
                score = 0.0
            else:
                # Weight by IDF
                score = 0.0
                for word_id in common:
                    word = self.db.id_to_word.get(word_id, "")
                    if word:
                        freq = self.db.word_frequencies.get(word, 1)
                        idf = math.log(self.db.total_definitions / max(freq, 1))
                        score += idf
            
            # Normalize
            if len(gloss_ids) > 0:
                score = score / math.log2(len(gloss_ids) + 1)
            
            # Frequency bias
            frequency_bias = 1.0 / (i + 1)
            final_score = score + (0.05 * frequency_bias)
            
            sense_scores.append((sense['id'], final_score, sense))
        
        # Sort and select best
        sense_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_id, best_score, best_sense = sense_scores[0]
        
        if best_score == 0:
            best_sense = senses[0]
            best_id = best_sense['id']
        
        return DisambiguatedWord(
            word=target_word,
            lemma=entry.get('lemma', target_word),
            selected_sense_id=best_id,
            selected_sense_definition=best_sense.get('definition', ''),
            confidence_score=best_score,
            all_sense_scores=[(s[0], s[1]) for s in sense_scores],
            context_window=context_words
        )


class DomainAwareWSD:
    """
    Domain-aware WSD that uses document domain information.
    
    If the document domain is known (e.g., medical, sports),
    senses with matching domains are preferred.
    """
    
    def __init__(self, database: LeskAr):
        self.db = database
        self.base_wsd = SimplifiedLesk(database)
    
    def disambiguate(
        self,
        target_word: str,
        context: str,
        document_domain: Optional[str] = None,
        window_size: int = 5
    ) -> Optional[DisambiguatedWord]:
        """
        Disambiguate with domain awareness.
        
        Args:
            target_word: The word to disambiguate
            context: The sentence containing the word
            document_domain: Known domain of the document
            window_size: Context window size
        
        Returns:
            DisambiguatedWord with the best sense
        """
        # First, get base disambiguation
        result = self.base_wsd.disambiguate(target_word, context, window_size)
        
        if not result or not document_domain:
            return result
        
        # Re-score with domain boost
        target_word = ArabicPreprocessor.normalize(target_word)
        entry = self.db.entries.get(target_word)
        
        if not entry:
            return result
        
        senses = entry.get('senses', [])
        
        # Boost scores for matching domains
        boosted_scores = []
        for sense_id, score in result.all_sense_scores:
            # Find sense
            sense = next(
                (s for s in senses if s['id'] == sense_id),
                None
            )
            
            if sense and sense.get('domain') == document_domain:
                score *= 1.5  # 50% boost for domain match
            
            boosted_scores.append((sense_id, score, sense))
        
        # Re-sort
        boosted_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_id, best_score, best_sense = boosted_scores[0]
        
        return DisambiguatedWord(
            word=result.word,
            lemma=result.lemma,
            selected_sense_id=best_id,
            selected_sense_definition=best_sense.get('definition', '') if best_sense else '',
            confidence_score=best_score,
            all_sense_scores=[(s[0], s[1]) for s in boosted_scores],
            context_window=result.context_window
        )


class ArabicWSD:
    """
    Main WSD class combining multiple strategies.
    
    Provides a unified interface for Arabic word sense disambiguation
    using DiLAC dictionary.
    """
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize WSD system.
        
        Args:
            database_path: Path to DiLAC-Lesk format JSON
        """
        self.db = LeskAr(database_path) if database_path else LeskAr()
        
        self.simplified_lesk = SimplifiedLesk(self.db)
        self.context_based = ContextBasedWSD(self.db)
        self.domain_aware = DomainAwareWSD(self.db)
    
    def load_database(self, filepath: str):
        """Load dictionary database"""
        self.db.load_database(filepath)
        self.simplified_lesk = SimplifiedLesk(self.db)
        self.context_based = ContextBasedWSD(self.db)
        self.domain_aware = DomainAwareWSD(self.db)
    
    def disambiguate(
        self,
        target_word: str,
        context: str,
        method: str = 'simplified_lesk',
        document_domain: Optional[str] = None,
        window_size: int = 5
    ) -> Optional[DisambiguatedWord]:
        """
        Disambiguate a word using the specified method.
        
        Args:
            target_word: The word to disambiguate
            context: The sentence/paragraph containing the word
            method: 'simplified_lesk', 'context_based', or 'domain_aware'
            document_domain: Document domain for domain-aware WSD
            window_size: Context window size
        
        Returns:
            DisambiguatedWord result
        """
        if method == 'simplified_lesk':
            return self.simplified_lesk.disambiguate(
                target_word, context, window_size
            )
        elif method == 'context_based':
            return self.context_based.disambiguate(
                target_word, context, window_size
            )
        elif method == 'domain_aware':
            return self.domain_aware.disambiguate(
                target_word, context, document_domain, window_size
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def disambiguate_text(
        self,
        text: str,
        method: str = 'simplified_lesk',
        document_domain: Optional[str] = None
    ) -> List[DisambiguatedWord]:
        """
        Disambiguate all content words in a text.
        
        Args:
            text: Input Arabic text
            method: WSD method to use
            document_domain: Optional document domain
        
        Returns:
            List of disambiguation results
        """
        results = []
        
        # Tokenize
        tokens = ArabicPreprocessor.tokenize(text, remove_stopwords=True)
        
        for token in tokens:
            normalized = ArabicPreprocessor.normalize(token)
            
            # Only disambiguate if word is in dictionary
            if normalized in self.db.entries:
                result = self.disambiguate(
                    normalized, text, method, document_domain
                )
                if result:
                    results.append(result)
        
        return results
    
    def get_word_senses(self, word: str) -> List[Dict]:
        """Get all senses of a word"""
        word = ArabicPreprocessor.normalize(word)
        entry = self.db.entries.get(word)
        
        if not entry:
            return []
        
        return entry.get('senses', [])


class WSDEvaluator:
    """Evaluator for WSD performance"""
    
    def __init__(self, wsd: ArabicWSD):
        self.wsd = wsd
    
    def evaluate(
        self,
        test_data: List[Dict],
        method: str = 'simplified_lesk'
    ) -> Dict:
        """
        Evaluate WSD on test data.
        
        Args:
            test_data: List of {'word', 'context', 'correct_sense_id'}
            method: WSD method to evaluate
        
        Returns:
            Evaluation metrics
        """
        correct = 0
        total = 0
        errors = []
        
        for item in test_data:
            result = self.wsd.disambiguate(
                item['word'],
                item['context'],
                method=method
            )
            
            if result:
                total += 1
                if result.selected_sense_id == item['correct_sense_id']:
                    correct += 1
                else:
                    errors.append({
                        'word': item['word'],
                        'context': item['context'],
                        'predicted': result.selected_sense_id,
                        'correct': item['correct_sense_id']
                    })
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'errors': errors
        }
    
    def compare_methods(
        self,
        test_data: List[Dict]
    ) -> Dict:
        """Compare all WSD methods"""
        methods = ['simplified_lesk', 'context_based', 'domain_aware']
        results = {}
        
        for method in methods:
            results[method] = self.evaluate(test_data, method)
        
        return results


if __name__ == "__main__":
    # Example usage
    wsd = ArabicWSD("data/processed/dilac_lesk.json")
    
    # Test disambiguation
    test_cases = [
        {
            'word': 'بنك',
            'context': 'ذهبت إلى البنك لسحب المال من حسابي',
            'expected_domain': 'اقتصاد'
        },
        {
            'word': 'بنك',
            'context': 'يقع البنك على ضفة النهر',
            'expected_domain': 'جغرافيا'
        },
    ]
    
    for test in test_cases:
        result = wsd.disambiguate(
            test['word'],
            test['context'],
            method='domain_aware',
            document_domain=test['expected_domain']
        )
        
        if result:
            print(f"\nWord: {test['word']}")
            print(f"Context: {test['context']}")
            print(f"Selected sense: {result.selected_sense_definition}")
            print(f"Confidence: {result.confidence_score:.4f}")
