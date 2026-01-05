"""
DiLAC: Dictionnaire de la Langue Arabe Contemporaine
=====================================================

A comprehensive Arabic lexical resource for semantic similarity
and word sense disambiguation.

Extended with Hybrid WSD capabilities combining:
- DiLAC dictionary knowledge (glosses, domains, examples)
- Transformer models (AraBERT, CAMeLBERT)

Modules:
--------
- lmf_schema: LMF-compliant data structures
- parser: Dictionary parsing utilities
- similarity: Lesk-ar semantic similarity
- wsd: Traditional WSD algorithms
- evaluation: Benchmarking and metrics
- database: Database management
- hybrid_wsd: Hybrid WSD with transformers (NEW)
- finetune: Fine-tuning utilities (NEW)

Quick Start:
------------
    # Traditional DiLAC WSD
    from dilac import ArabicWSD, LeskAr
    
    wsd = ArabicWSD('data/processed/dilac_lesk.json')
    result = wsd.disambiguate('بنك', 'ذهبت إلى البنك لسحب المال')
    
    # Hybrid WSD with AraBERT
    from dilac import HybridArabicWSD, HybridMethod
    
    hybrid = HybridArabicWSD(
        dilac_database_path='data/processed/dilac_lesk.json',
        transformer_model='arabert'
    )
    result = hybrid.disambiguate(
        'بنك', 
        'ذهبت إلى البنك لسحب المال',
        method=HybridMethod.EMBEDDING_FUSION
    )

Version: 2.0.0 (Hybrid Edition)
"""

__version__ = "2.0.0"
__author__ = "DiLAC Team"

# Core modules (from DiLAC v1.0)
from .lmf_schema import (
    LexicalResource,
    Lexicon,
    LexicalEntry,
    Lemma,
    WordForm,
    Sense,
    Definition,
    Context,
    SenseRelation,
    MorphologicalPattern
)

from .parser import (
    DiLACParser,
    DictionaryEntry,
    ParsedSense
)

from .similarity import (
    LeskAr,
    ArabicPreprocessor
)

from .wsd import (
    ArabicWSD,
    SimplifiedLesk,
    ContextBasedWSD,
    DomainAwareWSD,
    DisambiguatedWord,
    WSDEvaluator
)

from .evaluation import (
    SimilarityEvaluator,
    AWSSBenchmark
)

from .database import (
    DiLACDatabase,
    DiLACLeskDatabase
)

# Hybrid WSD modules (NEW in v2.0)
try:
    from .hybrid_wsd import (
        HybridArabicWSD,
        HybridMethod,
        HybridDisambiguationResult,
        TransformerBackend,
        DiLACGlossBERT,
        EmbeddingFusionWSD,
        TwoStageWSD,
        HybridWSDEvaluator
    )
    HYBRID_AVAILABLE = True
except ImportError as e:
    HYBRID_AVAILABLE = False
    import warnings
    warnings.warn(
        f"Hybrid WSD modules not available. Install transformers: "
        f"pip install transformers torch. Error: {e}"
    )

# Fine-tuning modules (NEW in v2.0)
try:
    from .finetune import (
        HybridWSDTrainer,
        TrainingConfig,
        DiLACTrainingDataGenerator,
        WSDTrainingSample,
        FineTunedHybridWSD
    )
    FINETUNE_AVAILABLE = True
except ImportError:
    FINETUNE_AVAILABLE = False

# Public API
__all__ = [
    # Version info
    '__version__',
    '__author__',
    'HYBRID_AVAILABLE',
    'FINETUNE_AVAILABLE',
    
    # LMF Schema
    'LexicalResource',
    'Lexicon', 
    'LexicalEntry',
    'Lemma',
    'WordForm',
    'Sense',
    'Definition',
    'Context',
    'SenseRelation',
    'MorphologicalPattern',
    
    # Parser
    'DiLACParser',
    'DictionaryEntry',
    'ParsedSense',
    
    # Similarity
    'LeskAr',
    'ArabicPreprocessor',
    
    # Traditional WSD
    'ArabicWSD',
    'SimplifiedLesk',
    'ContextBasedWSD',
    'DomainAwareWSD',
    'DisambiguatedWord',
    'WSDEvaluator',
    
    # Evaluation
    'SimilarityEvaluator',
    'AWSSBenchmark',
    
    # Database
    'DiLACDatabase',
    'DiLACLeskDatabase',
    
    # Hybrid WSD (conditional)
    'HybridArabicWSD',
    'HybridMethod',
    'HybridDisambiguationResult',
    'TransformerBackend',
    'DiLACGlossBERT',
    'EmbeddingFusionWSD',
    'TwoStageWSD',
    'HybridWSDEvaluator',
    
    # Fine-tuning (conditional)
    'HybridWSDTrainer',
    'TrainingConfig',
    'DiLACTrainingDataGenerator',
    'WSDTrainingSample',
    'FineTunedHybridWSD',
]


def get_info():
    """Print system information"""
    info = f"""
DiLAC System v{__version__}
{'=' * 40}
Core Modules:
  - LMF Schema: ✓
  - Parser: ✓
  - Similarity (Lesk-ar): ✓
  - WSD Algorithms: ✓
  - Evaluation: ✓
  - Database: ✓

Hybrid WSD (AraBERT/CAMeLBERT):
  - Available: {'✓' if HYBRID_AVAILABLE else '✗'}
  
Fine-tuning:
  - Available: {'✓' if FINETUNE_AVAILABLE else '✗'}

To enable hybrid features:
  pip install transformers torch

Supported transformer models:
  - arabert (aubmindlab/bert-base-arabertv2)
  - arabert-large (aubmindlab/bert-large-arabertv2)
  - camelbert-msa (CAMeL-Lab/bert-base-arabic-camelbert-msa)
  - camelbert-mix (CAMeL-Lab/bert-base-arabic-camelbert-mix)
  - marbert (UBC-NLP/MARBERT)
  - arbert (UBC-NLP/ARBERT)
"""
    print(info)
    return info
