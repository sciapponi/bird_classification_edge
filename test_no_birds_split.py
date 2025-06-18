#!/usr/bin/env python3
"""
Test script to verify no_birds dataset split fix
"""

import sys
sys.path.append('.')

from datasets.dataset_factory import create_no_birds_dataset

def test_no_birds_split():
    """Test that no_birds samples are properly split between train/val/test"""
    
    # Test parameters
    num_samples = 0  # Use all available
    no_birds_label = 8  # Assuming 8 bird classes
    pregenerated_dir = "augmented_dataset/no_birds"
    
    print("Testing no_birds dataset split fix...")
    print("="*50)
    
    # Test each subset
    subsets = ['training', 'validation', 'testing']
    results = {}
    
    for subset in subsets:
        print(f"\nTesting subset: {subset}")
        dataset = create_no_birds_dataset(
            num_samples=num_samples,
            no_birds_label=no_birds_label,
            esc50_dir="esc-50/ESC-50-master",
            bird_data_dir="bird_sound_dataset",
            allowed_bird_classes=["Bubo_bubo", "Certhia_familiaris"],
            subset=subset,
            target_sr=32000,
            clip_duration=3.0,
            esc50_no_bird_ratio=0.5,
            load_pregenerated=True,
            pregenerated_dir=pregenerated_dir
        )
        
        if dataset:
            results[subset] = len(dataset)
            print(f"  -> {subset}: {len(dataset)} samples")
        else:
            results[subset] = 0
            print(f"  -> {subset}: No dataset created")
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY:")
    print("="*50)
    
    total = sum(results.values())
    print(f"Training samples: {results.get('training', 0)}")
    print(f"Validation samples: {results.get('validation', 0)}")
    print(f"Test samples: {results.get('testing', 0)}")
    print(f"Total samples: {total}")
    
    # Calculate percentages
    if total > 0:
        train_pct = results.get('training', 0) / total * 100
        val_pct = results.get('validation', 0) / total * 100
        test_pct = results.get('testing', 0) / total * 100
        
        print(f"\nPercentages:")
        print(f"Training: {train_pct:.1f}%")
        print(f"Validation: {val_pct:.1f}%")
        print(f"Test: {test_pct:.1f}%")
        
        # Check if split is reasonable (should be around 70/15/15)
        if 65 <= train_pct <= 75 and 10 <= val_pct <= 20 and 10 <= test_pct <= 20:
            print(f"\n✅ SPLIT LOOKS GOOD! (approximately 70/15/15)")
        else:
            print(f"\n❌ SPLIT LOOKS WRONG! Expected ~70/15/15")
            
        # Verify no overlap by checking that total equals original
        expected_total = 836  # Known total from augmented_dataset/no_birds
        if total == expected_total:
            print(f"✅ TOTAL MATCHES: {total} = {expected_total}")
        else:
            print(f"❌ TOTAL MISMATCH: {total} ≠ {expected_total}")
    else:
        print("❌ NO SAMPLES FOUND!")

if __name__ == "__main__":
    test_no_birds_split() 