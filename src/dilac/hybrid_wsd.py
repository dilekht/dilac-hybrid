"""
DiLAC Hybrid WSD: Integrating AraBERT/CAMeLBERT with DiLAC
============================================================

This module extends the DiLAC WSD system with transformer-based 
contextual embeddings from AraBERT and CAMeLBERT.

Architecture:
    1. DiLAC provides: sense inventory, glosses, domain labels
    2. Transformers provide: contextual embeddings, gloss embeddings
    3. Hybrid scoring combines both signals

Three Hybrid Approaches:
    1. GlossBERT-style: Encode (context, gloss) pairs, classify
    2. Embedding Fusion: Combine DiLAC overlap with embedding similarity
    3. Two-Stage: DiLAC filters candidates, BERT re-ranks

Based on DiLAC system (Chapter 6) and recent work:
    - ArabGlossBERT (Al-Hajj & Jarrar, 2021)
    - Ensemble BERT for Arabic WSD (Djaidri et al., 2025)
"""

import math
import json
import logging
from typing import List, Dict, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# DiLAC imports (existing system)
from .wsd import ArabicWSD, SimplifiedLesk, DisambiguatedWord, Sense
from .similarity import LeskAr, ArabicPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridMethod(Enum):
    """Available hybrid WSD methods"""
    GLOSSBERT = "glossbert"           # Context-gloss pair classification
    EMBEDDING_FUSION = "fusion"       # Combine DiLAC + embedding scores
    TWO_STAGE = "two_stage"          # DiLAC filter + BERT rerank
    ENSEMBLE = "ensemble"             # Weighted ensemble of all methods


@dataclass
class HybridDisambiguationResult(DisambiguatedWord):
    """Extended result with hybrid scoring details"""
    dilac_score: float = 0.0
    bert_score: float = 0.0
    fusion_weights: Dict[str, float] = field(default_factory=dict)
    method_used: str = ""
    candidate_senses_considered: int = 0


class TransformerBackend:
    """
    Abstract interface for transformer models.
    Supports AraBERT, CAMeLBERT, and other Arabic BERT variants.
    """
    
    SUPPORTED_MODELS = {
        'arabert': 'aubmindlab/bert-base-arabertv2',
        'arabert-large': 'aubmindlab/bert-large-arabertv2',
        'camelbert-msa': 'CAMeL-Lab/bert-base-arabic-camelbert-msa',
        'camelbert-mix': 'CAMeL-Lab/bert-base-arabic-camelbert-mix',
        'arabic-bert': 'asafaya/bert-base-arabic',
        'marbert': 'UBC-NLP/MARBERT',
        'arbert': 'UBC-NLP/ARBERT',
    }
    
    def __init__(
        self,
        model_name: str = 'arabert',
        device: str = 'auto',
        max_length: int = 512
    ):
        """
        Initialize transformer backend.
        
        Args:
            model_name: Key from SUPPORTED_MODELS or HuggingFace model path
            device: 'cpu', 'cuda', or 'auto'
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.device = device
        
        # Lazy loading - don't load until needed
        self._initialized = False
    
    def _lazy_init(self):
        """Initialize model on first use"""
        if self._initialized:
            return
        
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            # Resolve model path
            model_path = self.SUPPORTED_MODELS.get(
                self.model_name, self.model_name
            )
            
            logger.info(f"Loading transformer model: {model_path}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            
            # Set device
            if self.device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.model.to(self.device)
            self.model.eval()
            
            self._initialized = True
            logger.info(f"Model loaded on {self.device}")
            
        except ImportError:
            raise ImportError(
                "Transformers library required. Install with: "
                "pip install transformers torch"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def encode(
        self,
        texts: Union[str, List[str]],
        pooling: str = 'cls'
    ) -> np.ndarray:
        """
        Encode text(s) to embeddings.
        
        Args:
            texts: Single text or list of texts
            pooling: 'cls' (CLS token), 'mean' (mean pooling), 'max' (max pooling)
        
        Returns:
            Embeddings array of shape (n_texts, hidden_size)
        """
        self._lazy_init()
        
        import torch
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
            
            if pooling == 'cls':
                embeddings = hidden_states[:, 0, :]
            elif pooling == 'mean':
                # Mean pooling with attention mask
                mask = inputs['attention_mask'].unsqueeze(-1)
                embeddings = (hidden_states * mask).sum(1) / mask.sum(1)
            elif pooling == 'max':
                embeddings = hidden_states.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling: {pooling}")
        
        return embeddings.cpu().numpy()
    
    def encode_pair(
        self,
        text_a: str,
        text_b: str,
        pooling: str = 'cls'
    ) -> np.ndarray:
        """
        Encode a pair of texts (e.g., context + gloss).
        Uses [CLS] text_a [SEP] text_b [SEP] format.
        
        Args:
            text_a: First text (e.g., context with target word)
            text_b: Second text (e.g., sense gloss/definition)
            pooling: Pooling strategy
        
        Returns:
            Combined embedding
        """
        self._lazy_init()
        
        import torch
        
        # Encode as pair
        inputs = self.tokenizer(
            text_a,
            text_b,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            if pooling == 'cls':
                embedding = outputs.last_hidden_state[:, 0, :]
            else:
                hidden = outputs.last_hidden_state
                mask = inputs['attention_mask'].unsqueeze(-1)
                embedding = (hidden * mask).sum(1) / mask.sum(1)
        
        return embedding.cpu().numpy()
    
    def similarity(
        self,
        text1: str,
        text2: str,
        metric: str = 'cosine'
    ) -> float:
        """
        Compute similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            metric: 'cosine' or 'euclidean'
        
        Returns:
            Similarity score (0-1 for cosine)
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        if metric == 'cosine':
            # Cosine similarity
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(emb1.flatten(), emb2.flatten()) / (norm1 * norm2))
        elif metric == 'euclidean':
            # Normalized euclidean distance (inverted to similarity)
            dist = np.linalg.norm(emb1 - emb2)
            return 1.0 / (1.0 + dist)
        else:
            raise ValueError(f"Unknown metric: {metric}")


class DiLACGlossBERT:
    """
    GlossBERT-style WSD using DiLAC sense inventory.
    
    Approach:
        1. For target word w in context C, get candidate senses from DiLAC
        2. Create (context, gloss) pairs for each sense
        3. Encode pairs with BERT
        4. Score based on [CLS] representation or similarity
    
    Reference: ArabGlossBERT (Al-Hajj & Jarrar, 2021)
    """
    
    def __init__(
        self,
        dilac_db: LeskAr,
        transformer: TransformerBackend,
        use_domain_hint: bool = True
    ):
        """
        Initialize GlossBERT-style WSD.
        
        Args:
            dilac_db: DiLAC database instance
            transformer: Transformer backend
            use_domain_hint: Include domain in gloss
        """
        self.db = dilac_db
        self.transformer = transformer
        self.use_domain_hint = use_domain_hint
        self.preprocessor = ArabicPreprocessor()
    
    def _create_context_gloss_pair(
        self,
        target_word: str,
        context: str,
        sense: Dict
    ) -> Tuple[str, str]:
        """
        Create (context, gloss) pair for encoding.
        
        Args:
            target_word: Target word (marked in context)
            context: Full context sentence
            sense: Sense dictionary from DiLAC
        
        Returns:
            (context_with_marker, gloss_text) tuple
        """
        # Mark target word in context with special tokens
        # This helps BERT focus on the target
        marked_context = context.replace(
            target_word,
            f"[TGT] {target_word} [/TGT]"
        )
        
        # Build gloss text
        gloss_parts = [sense.get('definition', '')]
        
        # Add domain hint if available
        if self.use_domain_hint and sense.get('domain'):
            gloss_parts.insert(0, f"({sense['domain']})")
        
        # Add examples if short gloss
        if len(sense.get('definition', '')) < 20:
            examples = sense.get('examples', [])[:1]
            if examples:
                gloss_parts.append(f"مثال: {examples[0]}")
        
        gloss_text = ' '.join(gloss_parts)
        
        return marked_context, gloss_text
    
    def disambiguate(
        self,
        target_word: str,
        context: str,
        top_k: int = 1
    ) -> Optional[HybridDisambiguationResult]:
        """
        Disambiguate using GlossBERT approach.
        
        Args:
            target_word: Word to disambiguate
            context: Context sentence
            top_k: Number of top senses to return info for
        
        Returns:
            HybridDisambiguationResult
        """
        target_word = ArabicPreprocessor.normalize(target_word)
        
        # Get senses from DiLAC
        entry = self.db.entries.get(target_word)
        if not entry:
            logger.warning(f"Word '{target_word}' not in DiLAC")
            return None
        
        senses = entry.get('senses', [])
        if not senses:
            return None
        
        # Score each sense
        sense_scores = []
        
        for sense in senses:
            # Create context-gloss pair
            ctx, gloss = self._create_context_gloss_pair(
                target_word, context, sense
            )
            
            # Encode pair and get score
            # Higher similarity = better match
            pair_embedding = self.transformer.encode_pair(ctx, gloss)
            
            # Also encode context and gloss separately for comparison
            ctx_emb = self.transformer.encode(ctx)
            gloss_emb = self.transformer.encode(gloss)
            
            # Cosine similarity between context and gloss
            cos_sim = self._cosine_similarity(ctx_emb, gloss_emb)
            
            sense_scores.append({
                'sense': sense,
                'score': cos_sim,
                'pair_embedding': pair_embedding
            })
        
        # Sort by score
        sense_scores.sort(key=lambda x: x['score'], reverse=True)
        
        best = sense_scores[0]
        
        return HybridDisambiguationResult(
            word=target_word,
            lemma=entry.get('lemma', target_word),
            selected_sense_id=best['sense']['id'],
            selected_sense_definition=best['sense'].get('definition', ''),
            confidence_score=best['score'],
            all_sense_scores=[
                (s['sense']['id'], s['score']) for s in sense_scores
            ],
            context_window=[],
            dilac_score=0.0,
            bert_score=best['score'],
            method_used='glossbert',
            candidate_senses_considered=len(senses)
        )
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        a = a.flatten()
        b = b.flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class EmbeddingFusionWSD:
    """
    Fusion approach combining DiLAC scores with embedding similarity.
    
    Approach:
        1. Get DiLAC overlap score (Lesk-ar)
        2. Get BERT embedding similarity score
        3. Combine with learned/tuned weights
    
    Formula:
        final_score = α * dilac_score + β * bert_score + γ * domain_match
    """
    
    def __init__(
        self,
        dilac_wsd: SimplifiedLesk,
        transformer: TransformerBackend,
        alpha: float = 0.4,
        beta: float = 0.5,
        gamma: float = 0.1
    ):
        """
        Initialize fusion WSD.
        
        Args:
            dilac_wsd: DiLAC SimplifiedLesk instance
            transformer: Transformer backend
            alpha: Weight for DiLAC score
            beta: Weight for BERT score
            gamma: Weight for domain match
        """
        self.dilac_wsd = dilac_wsd
        self.transformer = transformer
        self.db = dilac_wsd.db
        
        # Fusion weights (can be tuned)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def disambiguate(
        self,
        target_word: str,
        context: str,
        document_domain: Optional[str] = None,
        window_size: int = 5
    ) -> Optional[HybridDisambiguationResult]:
        """
        Disambiguate using score fusion.
        
        Args:
            target_word: Word to disambiguate
            context: Context sentence
            document_domain: Optional known domain
            window_size: Context window size
        
        Returns:
            HybridDisambiguationResult
        """
        target_word = ArabicPreprocessor.normalize(target_word)
        
        # Get senses
        entry = self.db.entries.get(target_word)
        if not entry:
            return None
        
        senses = entry.get('senses', [])
        if not senses:
            return None
        
        # Get DiLAC scores
        dilac_result = self.dilac_wsd.disambiguate(
            target_word, context, window_size
        )
        
        if not dilac_result:
            return None
        
        # Create score mapping from DiLAC
        dilac_scores = dict(dilac_result.all_sense_scores)
        
        # Normalize DiLAC scores to 0-1
        max_dilac = max(dilac_scores.values()) if dilac_scores else 1.0
        if max_dilac > 0:
            dilac_scores = {k: v/max_dilac for k, v in dilac_scores.items()}
        
        # Get BERT scores
        context_emb = self.transformer.encode(context)
        
        fused_scores = []
        
        for sense in senses:
            sense_id = sense['id']
            
            # DiLAC score
            d_score = dilac_scores.get(sense_id, 0.0)
            
            # BERT score: similarity between context and gloss
            gloss = sense.get('definition', '')
            if gloss:
                gloss_emb = self.transformer.encode(gloss)
                b_score = self._cosine_similarity(context_emb, gloss_emb)
            else:
                b_score = 0.0
            
            # Domain match score
            domain_score = 0.0
            if document_domain and sense.get('domain') == document_domain:
                domain_score = 1.0
            
            # Fused score
            final_score = (
                self.alpha * d_score +
                self.beta * b_score +
                self.gamma * domain_score
            )
            
            fused_scores.append({
                'sense': sense,
                'dilac_score': d_score,
                'bert_score': b_score,
                'domain_score': domain_score,
                'final_score': final_score
            })
        
        # Sort by final score
        fused_scores.sort(key=lambda x: x['final_score'], reverse=True)
        
        best = fused_scores[0]
        
        return HybridDisambiguationResult(
            word=target_word,
            lemma=entry.get('lemma', target_word),
            selected_sense_id=best['sense']['id'],
            selected_sense_definition=best['sense'].get('definition', ''),
            confidence_score=best['final_score'],
            all_sense_scores=[
                (s['sense']['id'], s['final_score']) for s in fused_scores
            ],
            context_window=dilac_result.context_window,
            dilac_score=best['dilac_score'],
            bert_score=best['bert_score'],
            fusion_weights={
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma
            },
            method_used='embedding_fusion',
            candidate_senses_considered=len(senses)
        )
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a = a.flatten()
        b = b.flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def tune_weights(
        self,
        validation_data: List[Dict],
        weight_grid: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Tune fusion weights on validation data.
        
        Args:
            validation_data: List of {'word', 'context', 'correct_sense_id'}
            weight_grid: Grid of weights to search
        
        Returns:
            Best weights
        """
        if weight_grid is None:
            weight_grid = {
                'alpha': [0.2, 0.3, 0.4, 0.5],
                'beta': [0.3, 0.4, 0.5, 0.6],
                'gamma': [0.0, 0.1, 0.2]
            }
        
        best_accuracy = 0.0
        best_weights = {'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma}
        
        for alpha in weight_grid['alpha']:
            for beta in weight_grid['beta']:
                for gamma in weight_grid['gamma']:
                    # Normalize weights
                    total = alpha + beta + gamma
                    if total == 0:
                        continue
                    
                    self.alpha = alpha / total
                    self.beta = beta / total
                    self.gamma = gamma / total
                    
                    # Evaluate
                    correct = 0
                    total_count = 0
                    
                    for item in validation_data:
                        result = self.disambiguate(
                            item['word'],
                            item['context'],
                            item.get('domain')
                        )
                        
                        if result:
                            total_count += 1
                            if result.selected_sense_id == item['correct_sense_id']:
                                correct += 1
                    
                    accuracy = correct / total_count if total_count > 0 else 0
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_weights = {
                            'alpha': self.alpha,
                            'beta': self.beta,
                            'gamma': self.gamma
                        }
        
        # Set best weights
        self.alpha = best_weights['alpha']
        self.beta = best_weights['beta']
        self.gamma = best_weights['gamma']
        
        logger.info(f"Best weights: {best_weights}, accuracy: {best_accuracy:.4f}")
        
        return best_weights


class TwoStageWSD:
    """
    Two-stage WSD: DiLAC filters, BERT re-ranks.
    
    Approach:
        1. Stage 1: Use DiLAC to get top-k candidate senses
        2. Stage 2: Use BERT to re-rank candidates
    
    This reduces BERT computation while leveraging DiLAC's efficiency.
    """
    
    def __init__(
        self,
        dilac_wsd: SimplifiedLesk,
        transformer: TransformerBackend,
        stage1_top_k: int = 3
    ):
        """
        Initialize two-stage WSD.
        
        Args:
            dilac_wsd: DiLAC WSD instance
            transformer: Transformer backend
            stage1_top_k: Number of candidates from stage 1
        """
        self.dilac_wsd = dilac_wsd
        self.transformer = transformer
        self.db = dilac_wsd.db
        self.stage1_top_k = stage1_top_k
    
    def disambiguate(
        self,
        target_word: str,
        context: str,
        window_size: int = 5
    ) -> Optional[HybridDisambiguationResult]:
        """
        Two-stage disambiguation.
        
        Args:
            target_word: Word to disambiguate
            context: Context sentence
            window_size: Context window for DiLAC
        
        Returns:
            HybridDisambiguationResult
        """
        target_word = ArabicPreprocessor.normalize(target_word)
        
        # Stage 1: DiLAC filtering
        dilac_result = self.dilac_wsd.disambiguate(
            target_word, context, window_size
        )
        
        if not dilac_result:
            return None
        
        # Get top-k candidates from DiLAC
        top_k_ids = [
            sense_id for sense_id, _ in dilac_result.all_sense_scores[:self.stage1_top_k]
        ]
        
        entry = self.db.entries.get(target_word)
        if not entry:
            return None
        
        senses = entry.get('senses', [])
        candidate_senses = [s for s in senses if s['id'] in top_k_ids]
        
        if not candidate_senses:
            # Fallback to DiLAC result
            return HybridDisambiguationResult(
                word=dilac_result.word,
                lemma=dilac_result.lemma,
                selected_sense_id=dilac_result.selected_sense_id,
                selected_sense_definition=dilac_result.selected_sense_definition,
                confidence_score=dilac_result.confidence_score,
                all_sense_scores=dilac_result.all_sense_scores,
                context_window=dilac_result.context_window,
                dilac_score=dilac_result.confidence_score,
                bert_score=0.0,
                method_used='two_stage_fallback',
                candidate_senses_considered=len(senses)
            )
        
        # Stage 2: BERT re-ranking
        context_emb = self.transformer.encode(context)
        
        reranked_scores = []
        
        for sense in candidate_senses:
            gloss = sense.get('definition', '')
            
            if gloss:
                gloss_emb = self.transformer.encode(gloss)
                bert_score = self._cosine_similarity(context_emb, gloss_emb)
            else:
                bert_score = 0.0
            
            # Get original DiLAC score
            dilac_score = dict(dilac_result.all_sense_scores).get(sense['id'], 0.0)
            
            reranked_scores.append({
                'sense': sense,
                'dilac_score': dilac_score,
                'bert_score': bert_score
            })
        
        # Re-rank by BERT score (primary) with DiLAC as tiebreaker
        reranked_scores.sort(
            key=lambda x: (x['bert_score'], x['dilac_score']),
            reverse=True
        )
        
        best = reranked_scores[0]
        
        return HybridDisambiguationResult(
            word=target_word,
            lemma=entry.get('lemma', target_word),
            selected_sense_id=best['sense']['id'],
            selected_sense_definition=best['sense'].get('definition', ''),
            confidence_score=best['bert_score'],
            all_sense_scores=[
                (s['sense']['id'], s['bert_score']) for s in reranked_scores
            ],
            context_window=dilac_result.context_window,
            dilac_score=best['dilac_score'],
            bert_score=best['bert_score'],
            method_used='two_stage',
            candidate_senses_considered=len(candidate_senses)
        )
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a = a.flatten()
        b = b.flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class HybridArabicWSD:
    """
    Main Hybrid WSD system combining DiLAC with AraBERT/CAMeLBERT.
    
    This is the primary interface for the hybrid WSD system.
    It builds on DiLAC and adds transformer capabilities.
    """
    
    def __init__(
        self,
        dilac_database_path: Optional[str] = None,
        transformer_model: str = 'arabert',
        device: str = 'auto'
    ):
        """
        Initialize hybrid WSD system.
        
        Args:
            dilac_database_path: Path to DiLAC-Lesk JSON database
            transformer_model: Transformer model name
            device: Computation device
        """
        # Initialize DiLAC components
        self.dilac = LeskAr(dilac_database_path) if dilac_database_path else LeskAr()
        self.simplified_lesk = SimplifiedLesk(self.dilac)
        
        # Initialize transformer (lazy loading)
        self.transformer = TransformerBackend(
            model_name=transformer_model,
            device=device
        )
        
        # Initialize hybrid methods
        self._glossbert = None
        self._fusion = None
        self._two_stage = None
        
        # Configuration
        self.default_method = HybridMethod.EMBEDDING_FUSION
    
    @property
    def glossbert(self) -> DiLACGlossBERT:
        """Lazy-load GlossBERT method"""
        if self._glossbert is None:
            self._glossbert = DiLACGlossBERT(self.dilac, self.transformer)
        return self._glossbert
    
    @property
    def fusion(self) -> EmbeddingFusionWSD:
        """Lazy-load Fusion method"""
        if self._fusion is None:
            self._fusion = EmbeddingFusionWSD(
                self.simplified_lesk, self.transformer
            )
        return self._fusion
    
    @property
    def two_stage(self) -> TwoStageWSD:
        """Lazy-load Two-Stage method"""
        if self._two_stage is None:
            self._two_stage = TwoStageWSD(
                self.simplified_lesk, self.transformer
            )
        return self._two_stage
    
    def load_database(self, filepath: str):
        """Load DiLAC database"""
        self.dilac.load_database(filepath)
        self.simplified_lesk = SimplifiedLesk(self.dilac)
        # Reset hybrid methods
        self._glossbert = None
        self._fusion = None
        self._two_stage = None
    
    def disambiguate(
        self,
        target_word: str,
        context: str,
        method: Union[str, HybridMethod] = None,
        document_domain: Optional[str] = None,
        window_size: int = 5
    ) -> Optional[HybridDisambiguationResult]:
        """
        Disambiguate a word using hybrid approach.
        
        Args:
            target_word: Word to disambiguate
            context: Context sentence
            method: Hybrid method to use (glossbert, fusion, two_stage, ensemble)
            document_domain: Optional document domain
            window_size: Context window size
        
        Returns:
            HybridDisambiguationResult
        """
        if method is None:
            method = self.default_method
        
        if isinstance(method, str):
            method = HybridMethod(method)
        
        if method == HybridMethod.GLOSSBERT:
            return self.glossbert.disambiguate(target_word, context)
        
        elif method == HybridMethod.EMBEDDING_FUSION:
            return self.fusion.disambiguate(
                target_word, context, document_domain, window_size
            )
        
        elif method == HybridMethod.TWO_STAGE:
            return self.two_stage.disambiguate(
                target_word, context, window_size
            )
        
        elif method == HybridMethod.ENSEMBLE:
            return self._ensemble_disambiguate(
                target_word, context, document_domain, window_size
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _ensemble_disambiguate(
        self,
        target_word: str,
        context: str,
        document_domain: Optional[str],
        window_size: int
    ) -> Optional[HybridDisambiguationResult]:
        """
        Ensemble of all methods with voting.
        """
        results = []
        
        # Get results from each method
        try:
            r1 = self.glossbert.disambiguate(target_word, context)
            if r1:
                results.append(('glossbert', r1))
        except Exception as e:
            logger.warning(f"GlossBERT failed: {e}")
        
        try:
            r2 = self.fusion.disambiguate(
                target_word, context, document_domain, window_size
            )
            if r2:
                results.append(('fusion', r2))
        except Exception as e:
            logger.warning(f"Fusion failed: {e}")
        
        try:
            r3 = self.two_stage.disambiguate(target_word, context, window_size)
            if r3:
                results.append(('two_stage', r3))
        except Exception as e:
            logger.warning(f"Two-stage failed: {e}")
        
        if not results:
            return None
        
        # Voting: weighted by confidence
        votes = {}
        for method_name, result in results:
            sense_id = result.selected_sense_id
            score = result.confidence_score
            
            if sense_id not in votes:
                votes[sense_id] = {'total_score': 0.0, 'count': 0, 'result': result}
            
            votes[sense_id]['total_score'] += score
            votes[sense_id]['count'] += 1
        
        # Select sense with highest total score
        best_sense_id = max(votes.keys(), key=lambda x: votes[x]['total_score'])
        best_result = votes[best_sense_id]['result']
        
        return HybridDisambiguationResult(
            word=best_result.word,
            lemma=best_result.lemma,
            selected_sense_id=best_sense_id,
            selected_sense_definition=best_result.selected_sense_definition,
            confidence_score=votes[best_sense_id]['total_score'] / len(results),
            all_sense_scores=best_result.all_sense_scores,
            context_window=best_result.context_window,
            dilac_score=best_result.dilac_score,
            bert_score=best_result.bert_score,
            method_used='ensemble',
            candidate_senses_considered=best_result.candidate_senses_considered
        )
    
    def disambiguate_text(
        self,
        text: str,
        method: Union[str, HybridMethod] = None,
        document_domain: Optional[str] = None
    ) -> List[HybridDisambiguationResult]:
        """
        Disambiguate all content words in a text.
        
        Args:
            text: Input Arabic text
            method: Hybrid method to use
            document_domain: Optional document domain
        
        Returns:
            List of disambiguation results
        """
        results = []
        
        tokens = ArabicPreprocessor.tokenize(text, remove_stopwords=True)
        
        for token in tokens:
            normalized = ArabicPreprocessor.normalize(token)
            
            if normalized in self.dilac.entries:
                result = self.disambiguate(
                    normalized, text, method, document_domain
                )
                if result:
                    results.append(result)
        
        return results
    
    def compare_methods(
        self,
        target_word: str,
        context: str,
        document_domain: Optional[str] = None
    ) -> Dict[str, HybridDisambiguationResult]:
        """
        Compare all hybrid methods on a single example.
        
        Args:
            target_word: Word to disambiguate
            context: Context sentence
            document_domain: Optional domain
        
        Returns:
            Dictionary of method -> result
        """
        comparison = {}
        
        for method in HybridMethod:
            try:
                result = self.disambiguate(
                    target_word, context, method, document_domain
                )
                if result:
                    comparison[method.value] = result
            except Exception as e:
                logger.warning(f"Method {method.value} failed: {e}")
        
        return comparison


class HybridWSDEvaluator:
    """Evaluator for hybrid WSD system"""
    
    def __init__(self, wsd: HybridArabicWSD):
        self.wsd = wsd
    
    def evaluate(
        self,
        test_data: List[Dict],
        method: Union[str, HybridMethod] = None
    ) -> Dict:
        """
        Evaluate hybrid WSD on test data.
        
        Args:
            test_data: List of {'word', 'context', 'correct_sense_id', 'domain'?}
            method: Method to evaluate
        
        Returns:
            Evaluation metrics
        """
        correct = 0
        total = 0
        dilac_scores = []
        bert_scores = []
        errors = []
        
        for item in test_data:
            result = self.wsd.disambiguate(
                item['word'],
                item['context'],
                method=method,
                document_domain=item.get('domain')
            )
            
            if result:
                total += 1
                dilac_scores.append(result.dilac_score)
                bert_scores.append(result.bert_score)
                
                if result.selected_sense_id == item['correct_sense_id']:
                    correct += 1
                else:
                    errors.append({
                        'word': item['word'],
                        'context': item['context'][:50] + '...',
                        'predicted': result.selected_sense_id,
                        'correct': item['correct_sense_id'],
                        'confidence': result.confidence_score
                    })
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_dilac_score': np.mean(dilac_scores) if dilac_scores else 0,
            'avg_bert_score': np.mean(bert_scores) if bert_scores else 0,
            'errors': errors[:10]  # First 10 errors
        }
    
    def compare_all_methods(
        self,
        test_data: List[Dict]
    ) -> Dict[str, Dict]:
        """Compare all hybrid methods"""
        results = {}
        
        for method in HybridMethod:
            logger.info(f"Evaluating {method.value}...")
            results[method.value] = self.evaluate(test_data, method)
        
        # Also evaluate pure DiLAC for comparison
        dilac_wsd = ArabicWSD()
        dilac_wsd.db = self.wsd.dilac
        dilac_wsd.simplified_lesk = self.wsd.simplified_lesk
        
        dilac_correct = 0
        dilac_total = 0
        
        for item in test_data:
            result = dilac_wsd.disambiguate(
                item['word'],
                item['context'],
                method='simplified_lesk'
            )
            if result:
                dilac_total += 1
                if result.selected_sense_id == item['correct_sense_id']:
                    dilac_correct += 1
        
        results['dilac_only'] = {
            'accuracy': dilac_correct / dilac_total if dilac_total > 0 else 0,
            'correct': dilac_correct,
            'total': dilac_total
        }
        
        return results


# Example usage and testing
if __name__ == "__main__":
    print("DiLAC Hybrid WSD System")
    print("=" * 50)
    
    # Initialize (without actual model loading for demo)
    print("\nTo use the hybrid system:")
    print("""
    from dilac.hybrid_wsd import HybridArabicWSD, HybridMethod
    
    # Initialize
    wsd = HybridArabicWSD(
        dilac_database_path='data/processed/dilac_lesk.json',
        transformer_model='arabert',  # or 'camelbert-msa'
        device='auto'
    )
    
    # Disambiguate
    result = wsd.disambiguate(
        target_word='بنك',
        context='ذهبت إلى البنك لسحب المال',
        method=HybridMethod.EMBEDDING_FUSION,
        document_domain='اقتصاد'
    )
    
    print(f"Selected sense: {result.selected_sense_definition}")
    print(f"DiLAC score: {result.dilac_score:.4f}")
    print(f"BERT score: {result.bert_score:.4f}")
    print(f"Combined confidence: {result.confidence_score:.4f}")
    
    # Compare methods
    comparison = wsd.compare_methods('بنك', 'ذهبت إلى البنك لسحب المال')
    for method, res in comparison.items():
        print(f"{method}: {res.selected_sense_id} ({res.confidence_score:.4f})")
    """)
