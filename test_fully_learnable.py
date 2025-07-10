#!/usr/bin/env python3
"""
Test script for Fully Learnable Filter Bank implementation.
This script tests the new FullyLearnableFilterBank class and ensures compatibility.
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append('.')

from differentiable_spec_torch import FullyLearnableFilterBank, create_spectrogram_module
from models import Improved_Phi_GRU_ATT

def test_fully_learnable_filter_bank():
    """Test the FullyLearnableFilterBank class directly."""
    print("=" * 60)
    print("Testing FullyLearnableFilterBank class")
    print("=" * 60)
    
    # Test parameters
    n_filters = 64
    n_freq_bins = 513
    sample_rate = 32000
    batch_size = 2
    audio_length = 96000  # 3 seconds at 32kHz
    
    # Test different initialization strategies
    for init_strategy in ['random', 'triangular_noise', 'xavier']:
        print(f"\n--- Testing initialization strategy: {init_strategy} ---")
        
        try:
            # Create filter bank
            filter_bank = FullyLearnableFilterBank(
                n_filters=n_filters,
                n_freq_bins=n_freq_bins,
                sample_rate=sample_rate,
                init_strategy=init_strategy
            )
            
            # Check parameter count
            total_params = sum(p.numel() for p in filter_bank.parameters())
            expected_params = n_filters * n_freq_bins
            print(f"  Parameter count: {total_params} (expected: {expected_params})")
            assert total_params == expected_params, f"Parameter count mismatch!"
            
            # Test forward pass
            dummy_audio = torch.randn(batch_size, audio_length)
            
            # Forward pass
            output = filter_bank(dummy_audio)
            expected_shape = (batch_size, n_filters, -1)  # Time dimension varies
            
            print(f"  Input shape: {dummy_audio.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Expected shape: {expected_shape[:-1]} + (time_frames,)")
            
            assert output.shape[0] == batch_size, "Batch dimension mismatch"
            assert output.shape[1] == n_filters, "Filter dimension mismatch"
            assert len(output.shape) == 3, "Should be 3D tensor"
            
            # Test gradient flow
            loss = output.sum()
            loss.backward()
            
            # Check if gradients exist
            assert filter_bank.filter_bank.grad is not None, "No gradients found"
            print(f"  Gradient norm: {filter_bank.filter_bank.grad.norm().item():.4f}")
            
            # Test single waveform input (no batch dimension)
            single_audio = torch.randn(audio_length)
            single_output = filter_bank(single_audio)
            print(f"  Single input shape: {single_audio.shape}")
            print(f"  Single output shape: {single_output.shape}")
            
            print(f"  ✅ {init_strategy} initialization test PASSED")
            
        except Exception as e:
            print(f"  ❌ {init_strategy} initialization test FAILED: {e}")
            raise

def test_factory_function():
    """Test the create_spectrogram_module factory function."""
    print("\n" + "=" * 60)
    print("Testing create_spectrogram_module factory function")
    print("=" * 60)
    
    # Test configurations for both types
    test_configs = [
        {
            'name': 'combined_log_linear',
            'config': {
                'spectrogram_type': 'combined_log_linear',
                'sample_rate': 32000,
                'n_linear_filters': 64,
                'n_fft': 1024,
                'hop_length': 320,
                'f_min': 150.0,
                'f_max': 16000.0,
                'initial_breakpoint': 4000.0,
                'initial_transition_width': 100.0,
                'trainable_filterbank': True
            }
        },
        {
            'name': 'fully_learnable',
            'config': {
                'spectrogram_type': 'fully_learnable',
                'sample_rate': 32000,
                'n_linear_filters': 64,
                'n_fft': 1024,
                'hop_length': 320,
                'filter_init_strategy': 'triangular_noise'
            }
        }
    ]
    
    batch_size = 1
    audio_length = 32000  # 1 second
    
    for test_case in test_configs:
        config_name = test_case['name']
        config = test_case['config']
        
        print(f"\n--- Testing {config_name} ---")
        
        try:
            # Create module using factory
            module = create_spectrogram_module(config)
            
            # Count parameters
            total_params = sum(p.numel() for p in module.parameters())
            print(f"  Total parameters: {total_params}")
            
            # Test forward pass
            dummy_audio = torch.randn(batch_size, audio_length)
            output = module(dummy_audio)
            
            print(f"  Input shape: {dummy_audio.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Module type: {type(module).__name__}")
            
            # Check output characteristics
            assert len(output.shape) == 3, "Output should be 3D"
            assert output.shape[0] == batch_size, "Batch dimension mismatch"
            assert output.shape[1] == 64, "Should have 64 filters"
            
            print(f"  ✅ {config_name} factory test PASSED")
            
        except Exception as e:
            print(f"  ❌ {config_name} factory test FAILED: {e}")
            raise

def test_model_integration():
    """Test integration with the main model."""
    print("\n" + "=" * 60)
    print("Testing Model Integration")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        {
            'name': 'Semi-Learnable',
            'spectrogram_type': 'combined_log_linear'
        },
        {
            'name': 'Fully Learnable',
            'spectrogram_type': 'fully_learnable'
        }
    ]
    
    batch_size = 2
    audio_length = 96000  # 3 seconds
    num_classes = 9
    
    for test_case in test_configs:
        config_name = test_case['name']
        spec_type = test_case['spectrogram_type']
        
        print(f"\n--- Testing {config_name} Model Integration ---")
        
        try:
            # Create model
            model = Improved_Phi_GRU_ATT(
                num_classes=num_classes,
                spectrogram_type=spec_type,
                sample_rate=32000,
                n_linear_filters=64,
                hidden_dim=32
            )
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            filter_params = 0
            
            # Count filter-specific parameters
            for name, param in model.named_parameters():
                if any(x in name for x in ['breakpoint', 'transition_width', 'filter_bank']):
                    filter_params += param.numel()
                    print(f"    Filter param: {name}, shape: {param.shape}, count: {param.numel()}")
            
            print(f"  Total model parameters: {total_params}")
            print(f"  Filter parameters: {filter_params}")
            print(f"  Other parameters: {total_params - filter_params}")
            
            # Test forward pass
            dummy_audio = torch.randn(batch_size, audio_length)
            model.eval()
            
            with torch.no_grad():
                output = model(dummy_audio)
            
            print(f"  Input shape: {dummy_audio.shape}")
            print(f"  Output shape: {output.shape}")
            
            # Check output
            assert output.shape == (batch_size, num_classes), f"Expected {(batch_size, num_classes)}, got {output.shape}"
            
            # Test training mode and backward pass
            model.train()
            output = model(dummy_audio)
            loss = output.sum()
            loss.backward()
            
            # Check gradients for filter parameters
            filter_grads_exist = False
            for name, param in model.named_parameters():
                if any(x in name for x in ['breakpoint', 'transition_width', 'filter_bank']):
                    if param.grad is not None:
                        filter_grads_exist = True
                        print(f"    Gradient for {name}: norm = {param.grad.norm().item():.4f}")
            
            assert filter_grads_exist, "No gradients found for filter parameters"
            
            print(f"  ✅ {config_name} model integration test PASSED")
            
        except Exception as e:
            print(f"  ❌ {config_name} model integration test FAILED: {e}")
            raise

def compare_parameter_counts():
    """Compare parameter counts between approaches."""
    print("\n" + "=" * 60)
    print("Parameter Count Comparison")
    print("=" * 60)
    
    configs = {
        'Semi-Learnable': 'combined_log_linear',
        'Fully Learnable': 'fully_learnable'
    }
    
    results = {}
    
    for name, spec_type in configs.items():
        model = Improved_Phi_GRU_ATT(
            num_classes=9,
            spectrogram_type=spec_type,
            sample_rate=32000,
            n_linear_filters=64,
            hidden_dim=32
        )
        
        total = sum(p.numel() for p in model.parameters())
        filter_params = sum(p.numel() for name, p in model.named_parameters() 
                           if any(x in name for x in ['breakpoint', 'transition_width', 'filter_bank']))
        other = total - filter_params
        
        results[name] = {
            'total': total,
            'filter': filter_params,
            'other': other
        }
        
        print(f"{name}:")
        print(f"  Total parameters: {total:,}")
        print(f"  Filter parameters: {filter_params:,}")
        print(f"  Other parameters: {other:,}")
        print(f"  Filter percentage: {filter_params/total*100:.2f}%")
    
    # Comparison
    print("\nComparison:")
    semi = results['Semi-Learnable']
    fully = results['Fully Learnable']
    
    print(f"  Parameter increase: {fully['total'] - semi['total']:,} (+{(fully['total']/semi['total']-1)*100:.1f}%)")
    print(f"  Filter param ratio: {fully['filter']/semi['filter']:.0f}x more")
    print(f"  Model size increase: {(fully['total']/semi['total']-1)*100:.1f}%")

if __name__ == "__main__":
    print("Testing Fully Learnable Filter Bank Implementation")
    print("=" * 80)
    
    try:
        test_fully_learnable_filter_bank()
        test_factory_function()
        test_model_integration()
        compare_parameter_counts()
        
        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED! Implementation is working correctly.")
        print("=" * 80)
        
        print("\nNext steps:")
        print("1. Run comparison experiments with config/comparison_config.yaml")
        print("2. Monitor filter evolution with config/fully_learnable_config.yaml")
        print("3. Compare performance and overfitting behavior")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 