"""
DiLAC Hybrid WSD: Fine-tuning Module
=====================================

This module provides fine-tuning capabilities for the hybrid WSD system.
It allows training BERT-based classifiers on DiLAC sense data.

Training Approaches:
    1. Context-Gloss Classification: Binary classification on (context, gloss) pairs
    2. Sense Classification: Multi-class classification for each word
    3. Contrastive Learning: Learn to distinguish correct vs incorrect senses

Based on:
    - ArabGlossBERT (Al-Hajj & Jarrar, 2021)
    - Ensemble BERT for Arabic WSD (Djaidri et al., 2025)
"""

import json
import logging
import os
import random
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WSDTrainingSample:
    """Training sample for WSD"""
    word: str
    context: str
    sense_id: str
    gloss: str
    domain: Optional[str]
    label: int  # 1 for correct sense, 0 for incorrect
    

@dataclass
class TrainingConfig:
    """Configuration for fine-tuning"""
    model_name: str = 'aubmindlab/bert-base-arabertv2'
    max_length: int = 256
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    negative_samples: int = 3  # Number of negative samples per positive
    output_dir: str = 'models/hybrid_wsd'
    seed: int = 42


class DiLACTrainingDataGenerator:
    """
    Generate training data from DiLAC for WSD fine-tuning.
    
    Creates context-gloss pairs with positive and negative examples.
    """
    
    def __init__(self, dilac_database: Dict):
        """
        Initialize data generator.
        
        Args:
            dilac_database: DiLAC entries dictionary
        """
        self.entries = dilac_database
        self.all_glosses = self._collect_all_glosses()
    
    def _collect_all_glosses(self) -> List[Tuple[str, str]]:
        """Collect all (word, gloss) pairs for negative sampling"""
        glosses = []
        for word, entry in self.entries.items():
            for sense in entry.get('senses', []):
                if sense.get('definition'):
                    glosses.append((word, sense['definition']))
        return glosses
    
    def generate_context_gloss_pairs(
        self,
        annotated_data: List[Dict],
        negative_samples: int = 3
    ) -> List[WSDTrainingSample]:
        """
        Generate training samples from annotated data.
        
        Args:
            annotated_data: List of {'word', 'context', 'correct_sense_id'}
            negative_samples: Number of negative examples per positive
        
        Returns:
            List of training samples
        """
        samples = []
        
        for item in annotated_data:
            word = item['word']
            context = item['context']
            correct_sense_id = item['correct_sense_id']
            
            entry = self.entries.get(word)
            if not entry:
                continue
            
            senses = entry.get('senses', [])
            
            # Positive sample: correct sense
            correct_sense = next(
                (s for s in senses if s['id'] == correct_sense_id),
                None
            )
            
            if correct_sense and correct_sense.get('definition'):
                samples.append(WSDTrainingSample(
                    word=word,
                    context=context,
                    sense_id=correct_sense_id,
                    gloss=correct_sense['definition'],
                    domain=correct_sense.get('domain'),
                    label=1
                ))
                
                # Negative samples: other senses of same word
                other_senses = [s for s in senses if s['id'] != correct_sense_id]
                
                for neg_sense in other_senses[:negative_samples]:
                    if neg_sense.get('definition'):
                        samples.append(WSDTrainingSample(
                            word=word,
                            context=context,
                            sense_id=neg_sense['id'],
                            gloss=neg_sense['definition'],
                            domain=neg_sense.get('domain'),
                            label=0
                        ))
                
                # Additional negative samples from random words
                remaining = negative_samples - len(other_senses)
                if remaining > 0:
                    random_glosses = random.sample(
                        self.all_glosses,
                        min(remaining, len(self.all_glosses))
                    )
                    for rand_word, rand_gloss in random_glosses:
                        if rand_word != word:
                            samples.append(WSDTrainingSample(
                                word=word,
                                context=context,
                                sense_id=f"random_{rand_word}",
                                gloss=rand_gloss,
                                domain=None,
                                label=0
                            ))
        
        return samples
    
    def generate_from_dilac(
        self,
        num_samples: int = 10000,
        context_window: int = 5
    ) -> List[WSDTrainingSample]:
        """
        Generate synthetic training data from DiLAC examples.
        
        Uses example sentences from DiLAC as contexts.
        
        Args:
            num_samples: Target number of samples
            context_window: Words around target in context
        
        Returns:
            List of training samples
        """
        samples = []
        
        # Collect entries with examples
        entries_with_examples = []
        for word, entry in self.entries.items():
            for sense in entry.get('senses', []):
                examples = sense.get('examples', [])
                if examples and sense.get('definition'):
                    entries_with_examples.append((word, entry, sense))
        
        if not entries_with_examples:
            logger.warning("No entries with examples found")
            return samples
        
        # Generate samples
        samples_per_entry = max(1, num_samples // len(entries_with_examples))
        
        for word, entry, sense in entries_with_examples:
            examples = sense.get('examples', [])
            
            for example in examples[:samples_per_entry]:
                # Positive sample
                samples.append(WSDTrainingSample(
                    word=word,
                    context=example,
                    sense_id=sense['id'],
                    gloss=sense['definition'],
                    domain=sense.get('domain'),
                    label=1
                ))
                
                # Negative: other senses
                other_senses = [
                    s for s in entry.get('senses', [])
                    if s['id'] != sense['id'] and s.get('definition')
                ]
                
                for neg in other_senses[:2]:
                    samples.append(WSDTrainingSample(
                        word=word,
                        context=example,
                        sense_id=neg['id'],
                        gloss=neg['definition'],
                        domain=neg.get('domain'),
                        label=0
                    ))
            
            if len(samples) >= num_samples:
                break
        
        return samples[:num_samples]


class HybridWSDTrainer:
    """
    Fine-tune BERT models for DiLAC hybrid WSD.
    
    Supports:
        1. Binary classification: Is this the correct sense?
        2. Contrastive learning: Pull correct sense closer, push incorrect away
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # Set seed for reproducibility
        random.seed(config.seed)
        np.random.seed(config.seed)
    
    def _init_model(self):
        """Initialize model and tokenizer"""
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                TrainingArguments,
                Trainer
            )
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=2  # Binary: correct/incorrect sense
            )
            
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
            self.model.to(self.device)
            
            logger.info(f"Model initialized on {self.device}")
            
        except ImportError:
            raise ImportError(
                "Required packages not found. Install with:\n"
                "pip install transformers torch datasets"
            )
    
    def prepare_dataset(
        self,
        samples: List[WSDTrainingSample],
        split_ratio: float = 0.9
    ) -> Tuple:
        """
        Prepare training and validation datasets.
        
        Args:
            samples: Training samples
            split_ratio: Train/val split ratio
        
        Returns:
            (train_dataset, val_dataset)
        """
        from datasets import Dataset
        
        # Shuffle samples
        random.shuffle(samples)
        
        # Create text pairs
        data = {
            'text_a': [],  # Context with marked target
            'text_b': [],  # Gloss
            'label': [],
            'word': [],
            'sense_id': []
        }
        
        for sample in samples:
            # Mark target word in context
            marked_context = sample.context.replace(
                sample.word,
                f"[TGT] {sample.word} [/TGT]"
            )
            
            # Add domain prefix to gloss if available
            gloss = sample.gloss
            if sample.domain:
                gloss = f"({sample.domain}) {gloss}"
            
            data['text_a'].append(marked_context)
            data['text_b'].append(gloss)
            data['label'].append(sample.label)
            data['word'].append(sample.word)
            data['sense_id'].append(sample.sense_id)
        
        # Split
        split_idx = int(len(samples) * split_ratio)
        
        train_data = {k: v[:split_idx] for k, v in data.items()}
        val_data = {k: v[split_idx:] for k, v in data.items()}
        
        train_dataset = Dataset.from_dict(train_data)
        val_dataset = Dataset.from_dict(val_data)
        
        return train_dataset, val_dataset
    
    def tokenize_function(self, examples):
        """Tokenize examples for training"""
        return self.tokenizer(
            examples['text_a'],
            examples['text_b'],
            padding='max_length',
            truncation=True,
            max_length=self.config.max_length
        )
    
    def train(
        self,
        train_samples: List[WSDTrainingSample],
        eval_samples: Optional[List[WSDTrainingSample]] = None
    ) -> Dict:
        """
        Fine-tune the model.
        
        Args:
            train_samples: Training samples
            eval_samples: Optional evaluation samples
        
        Returns:
            Training metrics
        """
        from transformers import TrainingArguments, Trainer
        from datasets import Dataset
        import torch
        
        self._init_model()
        
        # Prepare datasets
        if eval_samples:
            train_dataset = Dataset.from_dict({
                'text_a': [s.context.replace(s.word, f"[TGT] {s.word} [/TGT]") for s in train_samples],
                'text_b': [f"({s.domain}) {s.gloss}" if s.domain else s.gloss for s in train_samples],
                'label': [s.label for s in train_samples]
            })
            eval_dataset = Dataset.from_dict({
                'text_a': [s.context.replace(s.word, f"[TGT] {s.word} [/TGT]") for s in eval_samples],
                'text_b': [f"({s.domain}) {s.gloss}" if s.domain else s.gloss for s in eval_samples],
                'label': [s.label for s in eval_samples]
            })
        else:
            train_dataset, eval_dataset = self.prepare_dataset(train_samples)
        
        # Tokenize
        train_dataset = train_dataset.map(
            self.tokenize_function, batched=True
        )
        eval_dataset = eval_dataset.map(
            self.tokenize_function, batched=True
        )
        
        # Set format
        train_dataset.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'token_type_ids', 'label']
        )
        eval_dataset.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'token_type_ids', 'label']
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            seed=self.config.seed
        )
        
        # Compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            accuracy = (predictions == labels).mean()
            return {'accuracy': accuracy}
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics
        )
        
        # Train
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Evaluate
        eval_result = trainer.evaluate()
        
        # Save model
        trainer.save_model(f"{self.config.output_dir}/final")
        self.tokenizer.save_pretrained(f"{self.config.output_dir}/final")
        
        logger.info(f"Model saved to {self.config.output_dir}/final")
        
        return {
            'train_loss': train_result.training_loss,
            'eval_accuracy': eval_result['eval_accuracy'],
            'eval_loss': eval_result['eval_loss']
        }
    
    def save_for_inference(self, output_path: str):
        """Save model for inference"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        os.makedirs(output_path, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save config
        config_dict = {
            'model_name': self.config.model_name,
            'max_length': self.config.max_length,
            'num_labels': 2
        }
        
        with open(f"{output_path}/hybrid_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved for inference at {output_path}")


class FineTunedHybridWSD:
    """
    Use fine-tuned model for hybrid WSD.
    
    Loads a model trained with HybridWSDTrainer and uses it
    with DiLAC sense inventory for disambiguation.
    """
    
    def __init__(
        self,
        model_path: str,
        dilac_database: Dict
    ):
        """
        Initialize fine-tuned WSD.
        
        Args:
            model_path: Path to fine-tuned model
            dilac_database: DiLAC entries dictionary
        """
        self.model_path = model_path
        self.entries = dilac_database
        self.model = None
        self.tokenizer = None
        self.device = None
        
        self._load_model()
    
    def _load_model(self):
        """Load fine-tuned model"""
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer
        )
        import torch
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path
        )
        
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Load config
        config_path = f"{self.model_path}/hybrid_config.json"
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {'max_length': 256}
        
        logger.info(f"Fine-tuned model loaded from {self.model_path}")
    
    def disambiguate(
        self,
        target_word: str,
        context: str
    ) -> Optional[Dict]:
        """
        Disambiguate using fine-tuned model.
        
        Args:
            target_word: Word to disambiguate
            context: Context sentence
        
        Returns:
            Disambiguation result
        """
        import torch
        
        entry = self.entries.get(target_word)
        if not entry:
            return None
        
        senses = entry.get('senses', [])
        if not senses:
            return None
        
        # Mark target in context
        marked_context = context.replace(
            target_word,
            f"[TGT] {target_word} [/TGT]"
        )
        
        # Score each sense
        sense_scores = []
        
        for sense in senses:
            gloss = sense.get('definition', '')
            if sense.get('domain'):
                gloss = f"({sense['domain']}) {gloss}"
            
            # Tokenize
            inputs = self.tokenizer(
                marked_context,
                gloss,
                padding='max_length',
                truncation=True,
                max_length=self.config.get('max_length', 256),
                return_tensors='pt'
            ).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Probability of being correct sense (class 1)
                probs = torch.softmax(outputs.logits, dim=-1)
                score = probs[0, 1].item()
            
            sense_scores.append({
                'sense_id': sense['id'],
                'definition': sense.get('definition', ''),
                'domain': sense.get('domain'),
                'score': score
            })
        
        # Sort by score
        sense_scores.sort(key=lambda x: x['score'], reverse=True)
        
        best = sense_scores[0]
        
        return {
            'word': target_word,
            'selected_sense_id': best['sense_id'],
            'selected_sense_definition': best['definition'],
            'confidence': best['score'],
            'all_scores': sense_scores
        }


# Utility functions
def create_training_data_from_salma(
    salma_path: str,
    dilac_entries: Dict
) -> List[WSDTrainingSample]:
    """
    Create training data from SALMA corpus format.
    
    SALMA corpus structure: sense-annotated Arabic corpus
    
    Args:
        salma_path: Path to SALMA data
        dilac_entries: DiLAC entries for sense lookup
    
    Returns:
        Training samples
    """
    samples = []
    
    # Load SALMA data (adapt to actual format)
    with open(salma_path, 'r', encoding='utf-8') as f:
        salma_data = json.load(f)
    
    generator = DiLACTrainingDataGenerator(dilac_entries)
    
    # Convert SALMA format to our format
    annotated = []
    for item in salma_data:
        annotated.append({
            'word': item['target_word'],
            'context': item['sentence'],
            'correct_sense_id': item['sense_id']
        })
    
    samples = generator.generate_context_gloss_pairs(annotated)
    
    return samples


if __name__ == "__main__":
    print("DiLAC Hybrid WSD Fine-tuning Module")
    print("=" * 50)
    
    print("""
    Usage example:
    
    from dilac.finetune import (
        DiLACTrainingDataGenerator,
        HybridWSDTrainer,
        TrainingConfig,
        FineTunedHybridWSD
    )
    
    # 1. Generate training data from DiLAC
    with open('data/processed/dilac_lesk.json') as f:
        dilac_data = json.load(f)
    
    generator = DiLACTrainingDataGenerator(dilac_data['entries'])
    samples = generator.generate_from_dilac(num_samples=10000)
    
    # 2. Configure and train
    config = TrainingConfig(
        model_name='aubmindlab/bert-base-arabertv2',
        batch_size=16,
        num_epochs=3,
        output_dir='models/dilac_hybrid_wsd'
    )
    
    trainer = HybridWSDTrainer(config)
    metrics = trainer.train(samples)
    print(f"Training complete. Accuracy: {metrics['eval_accuracy']:.4f}")
    
    # 3. Use for inference
    wsd = FineTunedHybridWSD(
        model_path='models/dilac_hybrid_wsd/final',
        dilac_database=dilac_data['entries']
    )
    
    result = wsd.disambiguate('بنك', 'ذهبت إلى البنك لسحب المال')
    print(f"Selected: {result['selected_sense_definition']}")
    print(f"Confidence: {result['confidence']:.4f}")
    """)
