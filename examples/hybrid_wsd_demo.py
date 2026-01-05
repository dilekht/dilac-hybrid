#!/usr/bin/env python3
"""
DiLAC Hybrid WSD Example
========================

This example demonstrates how to use the hybrid WSD system
combining DiLAC with AraBERT/CAMeLBERT.

Prerequisites:
    pip install transformers torch

Usage:
    python examples/hybrid_wsd_demo.py
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def demo_basic_usage():
    """Demonstrate basic hybrid WSD usage"""
    print("\n" + "=" * 60)
    print("Demo 1: Basic Hybrid WSD Usage")
    print("=" * 60)
    
    try:
        from dilac import HybridArabicWSD, HybridMethod
        
        # Initialize (this will load the transformer model)
        print("\nInitializing Hybrid WSD system...")
        print("(First run will download AraBERT model ~500MB)")
        
        wsd = HybridArabicWSD(
            dilac_database_path='data/processed/dilac_lesk.json',
            transformer_model='arabert',
            device='auto'
        )
        
        # Test cases
        test_cases = [
            {
                'word': 'بنك',
                'context': 'ذهبت إلى البنك لسحب المال من حسابي',
                'expected': 'financial institution',
                'domain': 'اقتصاد'
            },
            {
                'word': 'بنك',
                'context': 'جلس الصياد على بنك النهر ينتظر السمك',
                'expected': 'river bank',
                'domain': 'جغرافيا'
            },
            {
                'word': 'عين',
                'context': 'تدمع العين من شدة الحزن والألم',
                'expected': 'eye (body part)',
                'domain': None
            },
            {
                'word': 'عين',
                'context': 'شربنا من عين الماء الصافية في الجبل',
                'expected': 'water spring',
                'domain': 'جغرافيا'
            }
        ]
        
        print("\nRunning disambiguation tests...")
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n--- Test {i} ---")
            print(f"Word: {test['word']}")
            print(f"Context: {test['context']}")
            print(f"Expected: {test['expected']}")
            
            result = wsd.disambiguate(
                target_word=test['word'],
                context=test['context'],
                method=HybridMethod.EMBEDDING_FUSION,
                document_domain=test['domain']
            )
            
            if result:
                print(f"Selected: {result.selected_sense_definition[:60]}...")
                print(f"DiLAC score: {result.dilac_score:.4f}")
                print(f"BERT score: {result.bert_score:.4f}")
                print(f"Combined: {result.confidence_score:.4f}")
            else:
                print("No result (word not in dictionary)")
        
        return wsd
        
    except ImportError as e:
        print(f"\nError: {e}")
        print("Please install transformers: pip install transformers torch")
        return None


def demo_method_comparison(wsd=None):
    """Compare all hybrid methods"""
    print("\n" + "=" * 60)
    print("Demo 2: Method Comparison")
    print("=" * 60)
    
    if wsd is None:
        print("Skipping (WSD not initialized)")
        return
    
    test_word = 'كتاب'
    test_context = 'قرأت الكتاب في المكتبة وأعجبني محتواه'
    
    print(f"\nWord: {test_word}")
    print(f"Context: {test_context}")
    print("\nComparing all methods:")
    
    comparison = wsd.compare_methods(test_word, test_context)
    
    print(f"\n{'Method':<20} {'Sense ID':<15} {'Confidence':<10}")
    print("-" * 50)
    
    for method, result in comparison.items():
        print(f"{method:<20} {result.selected_sense_id:<15} {result.confidence_score:.4f}")


def demo_batch_processing(wsd=None):
    """Demonstrate batch text processing"""
    print("\n" + "=" * 60)
    print("Demo 3: Batch Text Processing")
    print("=" * 60)
    
    if wsd is None:
        print("Skipping (WSD not initialized)")
        return
    
    from dilac import HybridMethod
    
    text = """
    ذهب الطالب إلى المكتبة لاستعارة كتاب عن التاريخ.
    ثم توجه إلى البنك لسحب بعض المال.
    في طريق العودة، جلس بجانب عين الماء ليستريح.
    """
    
    print(f"\nInput text:\n{text}")
    print("\nDisambiguating all content words...")
    
    results = wsd.disambiguate_text(
        text,
        method=HybridMethod.TWO_STAGE
    )
    
    print(f"\nFound {len(results)} disambiguated words:\n")
    
    for r in results:
        definition = r.selected_sense_definition
        if len(definition) > 50:
            definition = definition[:50] + "..."
        print(f"  {r.word}: {definition}")
        print(f"    Confidence: {r.confidence_score:.4f}")


def demo_without_transformers():
    """Demonstrate fallback to DiLAC-only when transformers unavailable"""
    print("\n" + "=" * 60)
    print("Demo 4: DiLAC-Only Mode (No Transformers)")
    print("=" * 60)
    
    try:
        from dilac import ArabicWSD
        
        print("\nUsing traditional DiLAC WSD (no transformer required)...")
        
        wsd = ArabicWSD('data/processed/dilac_lesk.json')
        
        result = wsd.disambiguate(
            target_word='بنك',
            context='ذهبت إلى البنك لسحب المال',
            method='simplified_lesk'
        )
        
        if result:
            print(f"\nWord: بنك")
            print(f"Selected: {result.selected_sense_definition[:60]}...")
            print(f"Confidence: {result.confidence_score:.4f}")
            print(f"Method: Simplified Lesk (DiLAC only)")
        
    except Exception as e:
        print(f"Error: {e}")


def demo_fine_tuning_workflow():
    """Demonstrate fine-tuning workflow (code only, doesn't run)"""
    print("\n" + "=" * 60)
    print("Demo 5: Fine-tuning Workflow (Code Example)")
    print("=" * 60)
    
    code = '''
# Fine-tuning DiLAC Hybrid WSD
# ============================

from dilac import (
    HybridWSDTrainer,
    TrainingConfig,
    DiLACTrainingDataGenerator,
    FineTunedHybridWSD
)
import json

# Step 1: Load DiLAC database
with open('data/processed/dilac_lesk.json') as f:
    dilac_data = json.load(f)

# Step 2: Generate training data from DiLAC examples
generator = DiLACTrainingDataGenerator(dilac_data['entries'])
samples = generator.generate_from_dilac(num_samples=10000)

print(f"Generated {len(samples)} training samples")

# Step 3: Configure training
config = TrainingConfig(
    model_name='aubmindlab/bert-base-arabertv2',
    batch_size=16,
    num_epochs=3,
    learning_rate=2e-5,
    output_dir='models/dilac_hybrid_wsd'
)

# Step 4: Train
trainer = HybridWSDTrainer(config)
metrics = trainer.train(samples)

print(f"Training complete!")
print(f"Accuracy: {metrics['eval_accuracy']:.4f}")
print(f"Loss: {metrics['eval_loss']:.4f}")

# Step 5: Use fine-tuned model
wsd = FineTunedHybridWSD(
    model_path='models/dilac_hybrid_wsd/final',
    dilac_database=dilac_data['entries']
)

result = wsd.disambiguate('بنك', 'ذهبت إلى البنك لسحب المال')
print(f"Selected: {result['selected_sense_definition']}")
print(f"Confidence: {result['confidence']:.4f}")
'''
    
    print("\nFine-tuning code example:")
    print("-" * 40)
    print(code)


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("DiLAC Hybrid WSD System - Demo")
    print("=" * 60)
    
    # Check system info
    try:
        from dilac import get_info, HYBRID_AVAILABLE
        get_info()
    except ImportError:
        print("DiLAC package not properly installed")
        return
    
    # Run demos
    wsd = demo_basic_usage()
    demo_method_comparison(wsd)
    demo_batch_processing(wsd)
    demo_without_transformers()
    demo_fine_tuning_workflow()
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
