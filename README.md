# DiLAC Hybrid WSD System

## Integrating AraBERT/CAMeLBERT with DiLAC for Arabic Word Sense Disambiguation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This system extends the DiLAC (Dictionnaire de la Langue Arabe Contemporaine) lexical resource with transformer-based contextual embeddings for improved Arabic Word Sense Disambiguation.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DiLAC Hybrid WSD System                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐         ┌─────────────────────────────┐   │
│  │     DiLAC       │         │   Transformer Backend       │   │
│  │   Database      │         │  (AraBERT/CAMeLBERT)        │   │
│  ├─────────────────┤         ├─────────────────────────────┤   │
│  │ • 32,300 entries│         │ • Contextual embeddings     │   │
│  │ • 63,019 senses │         │ • Context-gloss encoding    │   │
│  │ • 22 domains    │         │ • Semantic similarity       │   │
│  │ • Glosses       │         │                             │   │
│  │ • Examples      │         │                             │   │
│  └────────┬────────┘         └──────────────┬──────────────┘   │
│           │                                  │                  │
│           └──────────────┬───────────────────┘                  │
│                          ▼                                      │
│           ┌──────────────────────────────┐                      │
│           │      Hybrid Methods          │                      │
│           ├──────────────────────────────┤                      │
│           │ 1. GlossBERT-style           │                      │
│           │ 2. Embedding Fusion          │                      │
│           │ 3. Two-Stage (Filter+Rerank) │                      │
│           │ 4. Ensemble                  │                      │
│           └──────────────────────────────┘                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### DiLAC Foundation
- **Sense Inventory**: 63,019 senses from comprehensive Arabic dictionary
- **Rich Glosses**: Detailed definitions for context-gloss matching
- **Domain Labels**: 22 semantic domains for domain-aware disambiguation
- **Usage Examples**: 43,384 examples for training and evaluation

### Transformer Integration
- **AraBERT**: Arabic-specific BERT model
- **CAMeLBERT**: Multi-dialect Arabic BERT (MSA, Mix variants)
- **MARBERT/ARBERT**: Additional Arabic transformer options

### Hybrid Approaches
1. **GlossBERT-style**: Encode (context, gloss) pairs for binary classification
2. **Embedding Fusion**: Combine DiLAC overlap scores with embedding similarity
3. **Two-Stage**: DiLAC filters candidates, BERT re-ranks
4. **Ensemble**: Weighted voting across all methods

## Installation

```bash
# Clone repository
git clone https://github.com/dilekht/dilac-hybrid.git
cd dilac-hybrid

# Install core dependencies
pip install -r requirements.txt

# Install transformer dependencies (optional but recommended)
pip install transformers torch

# Install in development mode
pip install -e .
```

## Requirements

```
# Core (requirements.txt)
numpy>=1.20.0
scipy>=1.7.0

# Transformers (optional)
transformers>=4.20.0
torch>=1.10.0
datasets>=2.0.0

# Development
pytest>=7.0.0
```

## Quick Start

### 1. Basic Hybrid WSD

```python
from dilac import HybridArabicWSD, HybridMethod

# Initialize with DiLAC database and transformer
wsd = HybridArabicWSD(
    dilac_database_path='data/processed/dilac_lesk.json',
    transformer_model='arabert',  # or 'camelbert-msa'
    device='auto'  # 'cpu', 'cuda', or 'auto'
)

# Disambiguate using fusion method
result = wsd.disambiguate(
    target_word='بنك',
    context='ذهبت إلى البنك لسحب المال من حسابي',
    method=HybridMethod.EMBEDDING_FUSION,
    document_domain='اقتصاد'  # Optional domain hint
)

print(f"Selected sense: {result.selected_sense_definition}")
print(f"DiLAC score: {result.dilac_score:.4f}")
print(f"BERT score: {result.bert_score:.4f}")
print(f"Combined confidence: {result.confidence_score:.4f}")
```

### 2. Compare All Methods

```python
# Compare all hybrid methods on a single example
comparison = wsd.compare_methods(
    target_word='عين',
    context='تدمع العين من شدة الحزن',
    document_domain=None
)

for method, result in comparison.items():
    print(f"{method}: {result.selected_sense_id} "
          f"(conf: {result.confidence_score:.4f})")
```

### 3. Batch Processing

```python
# Disambiguate all words in a text
text = "ذهب الرجل إلى طبيب العيون بعدها زار مكان جميل حيث عيون الماء"

results = wsd.disambiguate_text(
    text,
    method=HybridMethod.TWO_STAGE
)

for r in results:
    print(f"{r.word}: {r.selected_sense_definition[:50]}...")
```

### 4. Fine-tuning (Optional)

```python
from dilac import (
    HybridWSDTrainer,
    TrainingConfig,
    DiLACTrainingDataGenerator
)
import json

# Load DiLAC database
with open('data/processed/dilac_lesk.json') as f:
    dilac_data = json.load(f)

# Generate training data from DiLAC examples
generator = DiLACTrainingDataGenerator(dilac_data['entries'])
samples = generator.generate_from_dilac(num_samples=10000)

# Configure training
config = TrainingConfig(
    model_name='aubmindlab/bert-base-arabertv2',
    batch_size=16,
    num_epochs=3,
    learning_rate=2e-5,
    output_dir='models/dilac_hybrid_wsd'
)

# Train
trainer = HybridWSDTrainer(config)
metrics = trainer.train(samples)

print(f"Final accuracy: {metrics['eval_accuracy']:.4f}")
```

## Hybrid Methods Explained

### 1. GlossBERT-style (`HybridMethod.GLOSSBERT`)

Encodes context and sense gloss as a pair:
```
[CLS] context with [TGT] word [/TGT] [SEP] (domain) definition [SEP]
```
Computes similarity based on the [CLS] representation.

**Best for**: When you want pure neural approach with DiLAC glosses.

### 2. Embedding Fusion (`HybridMethod.EMBEDDING_FUSION`)

Combines DiLAC Lesk-ar score with BERT embedding similarity:
```
final_score = α × dilac_score + β × bert_score + γ × domain_match
```

Default weights: α=0.4, β=0.5, γ=0.1

**Best for**: Balanced approach leveraging both methods.

### 3. Two-Stage (`HybridMethod.TWO_STAGE`)

1. **Stage 1**: DiLAC filters to top-k candidates (efficient)
2. **Stage 2**: BERT re-ranks candidates (accurate)

**Best for**: Large sense inventories where full BERT scoring is expensive.

### 4. Ensemble (`HybridMethod.ENSEMBLE`)

Weighted voting across all methods based on confidence scores.

**Best for**: Maximum accuracy when computation cost is acceptable.

## Supported Transformer Models

| Model | Key | HuggingFace Path |
|-------|-----|------------------|
| AraBERT v2 | `arabert` | `aubmindlab/bert-base-arabertv2` |
| AraBERT Large | `arabert-large` | `aubmindlab/bert-large-arabertv2` |
| CAMeLBERT MSA | `camelbert-msa` | `CAMeL-Lab/bert-base-arabic-camelbert-msa` |
| CAMeLBERT Mix | `camelbert-mix` | `CAMeL-Lab/bert-base-arabic-camelbert-mix` |
| MARBERT | `marbert` | `UBC-NLP/MARBERT` |
| ARBERT | `arbert` | `UBC-NLP/ARBERT` |

## Evaluation

```python
from dilac import HybridWSDEvaluator

# Prepare test data
test_data = [
    {'word': 'عين', 'context': '...', 'correct_sense_id': 'sense_1'},
    # ...
]

# Evaluate
evaluator = HybridWSDEvaluator(wsd)
results = evaluator.compare_all_methods(test_data)

# Print comparison
for method, metrics in results.items():
    print(f"{method}: {metrics['accuracy']:.4f}")
```

## Project Structure

```
dilac-hybrid/
├── src/dilac/
│   ├── __init__.py          # Package init with all exports
│   ├── lmf_schema.py        # LMF data structures
│   ├── parser.py            # Dictionary parser
│   ├── similarity.py        # Lesk-ar similarity
│   ├── wsd.py               # Traditional WSD algorithms
│   ├── evaluation.py        # Benchmarking
│   ├── database.py          # Database management
│   ├── hybrid_wsd.py        # Hybrid WSD (NEW)
│   └── finetune.py          # Fine-tuning utilities (NEW)
├── data/
│   ├── raw/                 # Raw dictionary files
│   ├── processed/           # Processed DiLAC databases
│   └── benchmarks/          # AWSS and other benchmarks
├── models/                  # Trained models
├── examples/                # Usage examples
├── tests/                   # Unit tests
├── requirements.txt
├── setup.py
└── README.md
```

## Performance Comparison

Expected results on AWSS benchmark (40 pairs):

| Method | MSE ↓ | Correlation ↑ |
|--------|-------|---------------|
| DiLAC Lesk-ar (baseline) | 0.0203 | 0.917 |
| GlossBERT-style | ~0.018 | ~0.93 |
| Embedding Fusion | ~0.016 | ~0.94 |
| Two-Stage | ~0.017 | ~0.93 |
| Ensemble | ~0.015 | ~0.95 |

*Note: Results may vary based on model and hyperparameters.*

## Citation

If you use this system, please cite:

```bibtex
@article{dilac2026hybrid,
  title={DiLAC: A Comprehensive Arabic Lexical Resource for Semantic 
         Similarity and Word Sense Disambiguation},
  author={Tahar Dilekh},
  journal={ACM Transactions on Asian and Low-Resource Language 
           Information Processing},
  year={2026}
}
```

## References

- Al-Hajj, M., & Jarrar, M. (2021). ArabGlossBERT: Fine-tuning BERT on Context-Gloss Pairs for WSD. RANLP 2021.
- Djaidri, A., et al. (2025). Enhancing Arabic Word Sense Disambiguation with Ensemble BERT-Based Models. ICALP 2024.
- Antoun, W., et al. (2020). AraBERT: Transformer-based Model for Arabic Language Understanding.
- Inoue, G., et al. (2021). CAMeLBERT: A Collection of Pre-trained Models for Arabic NLP.

## License

MIT License - see [LICENSE](LICENSE) file.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
