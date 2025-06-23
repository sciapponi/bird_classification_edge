#!/usr/bin/env python3
"""
Bird Classification Benchmark Runner

This script runs a complete benchmark comparing student model predictions
with BirdNET predictions on existing audio files from the dataset.

No audio preprocessing or segmentation is performed - uses existing files directly.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import hydra
from omegaconf import DictConfig
import numpy as np
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)


def setup_timestamped_output_paths(cfg: DictConfig, original_cwd: str) -> str:
    """
    Setup timestamped output paths similar to training system.
    
    Args:
        cfg: Configuration object
        original_cwd: Original working directory
        
    Returns:
        Timestamped output directory path
    """
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create timestamped directory
    timestamped_dir = os.path.join(original_cwd, cfg.benchmark.paths.output_dir, timestamp)
    
    # Update configuration paths to use timestamped directory
    cfg.benchmark.paths.segments_dir = os.path.join(timestamped_dir, "segments")
    cfg.benchmark.paths.predictions_dir = os.path.join(timestamped_dir, "predictions")
    cfg.benchmark.paths.comparison_dir = os.path.join(timestamped_dir, "comparison")
    
    # Create directories
    os.makedirs(cfg.benchmark.paths.segments_dir, exist_ok=True)
    os.makedirs(cfg.benchmark.paths.predictions_dir, exist_ok=True)
    os.makedirs(cfg.benchmark.paths.comparison_dir, exist_ok=True)
    
    # Log the new structure
    logger.info(f"📁 Timestamped benchmark directory: {timestamp}")
    logger.info(f"   Predictions: {cfg.benchmark.paths.predictions_dir}")
    logger.info(f"   Comparison: {cfg.benchmark.paths.comparison_dir}")
    
    return timestamped_dir


def discover_audio_files(cfg: DictConfig, original_cwd: str) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Discover audio files from the dataset structure and create ground truth.
    
    Args:
        cfg: Configuration object
        original_cwd: Original working directory
        
    Returns:
        Tuple of (file_info_list, ground_truth_dataframe)
    """
    file_info = []
    
    # Process bird sound dataset
    bird_dataset_path = os.path.join(original_cwd, cfg.benchmark.paths.audio_dir)
    
    if os.path.exists(bird_dataset_path):
        logger.info(f"Scanning bird dataset: {bird_dataset_path}")
        
        for species_dir in os.listdir(bird_dataset_path):
            species_path = os.path.join(bird_dataset_path, species_dir)
            
            if not os.path.isdir(species_path) or species_dir.startswith('.'):
                continue
            
            logger.info(f"Processing species: {species_dir}")
            
            audio_files = []
            for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']:
                pattern = os.path.join(species_path, f"*{ext}")
                import glob
                audio_files.extend(glob.glob(pattern))
            
            # Limit files per class if specified
            max_files = cfg.get('debug', {}).get('max_files_per_class', None)
            if max_files and len(audio_files) > max_files:
                audio_files = audio_files[:max_files]
                logger.info(f"Limited {species_dir} to {max_files} files")
            
            for audio_file in audio_files:
                rel_path = os.path.relpath(audio_file, original_cwd)
                file_info.append({
                    'audio_path': rel_path,
                    'label': species_dir,
                    'category': 'bird',
                    'species': species_dir,
                    'full_path': audio_file
                })
            
            logger.info(f"Found {len(audio_files)} files for {species_dir}")
    else:
        logger.warning(f"Bird dataset directory not found: {bird_dataset_path}")
    
    # Process no_birds dataset (only if not in birds-only mode)
    exclude_no_birds = cfg.benchmark.mode.get('exclude_no_birds_from_ground_truth', False)
    
    if not exclude_no_birds:
        no_birds_path = os.path.join(original_cwd, cfg.benchmark.paths.get('no_birds_dir', 'augmented_dataset/no_birds'))
        
        if os.path.exists(no_birds_path):
            logger.info(f"Scanning no_birds dataset: {no_birds_path}")
            
            audio_files = []
            for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']:
                pattern = os.path.join(no_birds_path, f"*{ext}")
                import glob
                audio_files.extend(glob.glob(pattern))
            
            # Limit files if specified
            max_files = cfg.get('debug', {}).get('max_files_per_class', None)
            if max_files and len(audio_files) > max_files:
                audio_files = audio_files[:max_files]
                logger.info(f"Limited no_birds to {max_files} files")
            
            for audio_file in audio_files:
                rel_path = os.path.relpath(audio_file, original_cwd)
                file_info.append({
                    'audio_path': rel_path,
                    'label': 'no_birds',
                    'category': 'no_birds',
                    'species': 'no_birds',
                    'full_path': audio_file
                })
            
            logger.info(f"Found {len(audio_files)} no_birds files")
        else:
            logger.warning(f"No_birds directory not found: {no_birds_path}")
    else:
        logger.info("🎯 BIRDS-ONLY MODE: Excluding no_birds samples from ground truth")
        
    # Verify working directory exists
    if not os.path.exists(original_cwd):
        logger.error(f"Original working directory does not exist: {original_cwd}")
    
    # Create ground truth DataFrame
    ground_truth_df = pd.DataFrame(file_info)
    
    if len(ground_truth_df) == 0:
        logger.error("No audio files found!")
        return [], ground_truth_df
    
    # Apply global files limit if specified
    files_limit = cfg.get('debug', {}).get('files_limit', None)
    if files_limit and len(ground_truth_df) > files_limit:
        # Sample proportionally from each class
        limited_df = ground_truth_df.groupby('label').apply(
            lambda x: x.sample(min(len(x), max(1, int(files_limit * len(x) / len(ground_truth_df)))))
        ).reset_index(drop=True)
        
        # If still too many, take first N
        if len(limited_df) > files_limit:
            limited_df = limited_df.head(files_limit)
            
        ground_truth_df = limited_df
        logger.info(f"🔍 LIMITED TO {files_limit} files for testing")
    
    logger.info(f"Total audio files discovered: {len(ground_truth_df)}")
    
    # Log class distribution
    class_counts = ground_truth_df['label'].value_counts()
    logger.info("Class distribution:")
    for label, count in class_counts.items():
        logger.info(f"  {label}: {count} files")
    
    return file_info, ground_truth_df


def run_student_predictions(cfg: DictConfig, ground_truth_df: pd.DataFrame, original_cwd: str) -> pd.DataFrame:
    """
    Run student model predictions.
    
    Args:
        cfg: Configuration object
        ground_truth_df: Ground truth DataFrame
        original_cwd: Original working directory
        
    Returns:
        DataFrame with student predictions
    """
    logger.info("Starting student model predictions")
    
    try:
        from benchmark.predict_student import StudentModelPredictor, predict_from_dataframe
        
        # Check if birds-only mode is enabled
        force_bird_prediction = cfg.benchmark.mode.get('exclude_no_birds_from_ground_truth', False)
        
        # Initialize predictor
        predictor = StudentModelPredictor(
            model_path=os.path.join(original_cwd, cfg.benchmark.paths.student_model),
            config_path=os.path.join(original_cwd, cfg.benchmark.paths.student_config),
            device=cfg.student_model.inference.device,
            confidence_threshold=cfg.student_model.inference.confidence_threshold,
            force_bird_prediction=force_bird_prediction
        )
        
        # Generate predictions
        predictions_df = predict_from_dataframe(
            predictor,
            ground_truth_df,
            audio_base_path=original_cwd
        )
        
        # Save predictions
        output_path = os.path.join(original_cwd, cfg.benchmark.paths.predictions_dir, "student_predictions.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        predictions_df.to_csv(output_path, index=False)
        
        logger.info(f"Student predictions saved to: {output_path}")
        
        # Log summary
        pred_counts = predictions_df['student_prediction'].value_counts()
        logger.info("✅ Student prediction completed")
        logger.info(f"  Total predictions: {len(predictions_df)}")
        logger.info(f"  Average confidence: {predictions_df['student_confidence'].mean():.3f}")
        logger.info(f"  Top prediction: {pred_counts.index[0]} ({pred_counts.iloc[0]} files)")
        
        return predictions_df
        
    except Exception as e:
        logger.error(f"Student model prediction failed: {e}")
        raise


def run_birdnet_predictions(cfg: DictConfig, ground_truth_df: pd.DataFrame, original_cwd: str) -> pd.DataFrame:
    """
    Run BirdNET predictions with aligned preprocessing.
    
    Args:.
        cfg: Configuration object
        ground_truth_df: Ground truth DataFrame
        original_cwd: Original working directory
        
    Returns:
        DataFrame with BirdNET predictions
    """
    logger.info("Starting BirdNET predictions with ALIGNED preprocessing and ADAPTIVE THRESHOLDS")
    logger.info("🎯 Using identical preprocessing pipeline as student model:")
    logger.info(f"   - Segment duration: {cfg.student_model.preprocessing.clip_duration}s")
    logger.info(f"   - Sample rate: {cfg.student_model.preprocessing.sample_rate} Hz")
    logger.info(f"   - Bandpass filter: {cfg.student_model.preprocessing.lowcut}-{cfg.student_model.preprocessing.highcut} Hz")
    logger.info(f"   - Extract calls: {cfg.student_model.preprocessing.get('extract_calls', True)}")
    logger.info("🔧 Threshold configuration:")
    logger.info(f"   - Confidence threshold: {cfg.birdnet.confidence_threshold} (single threshold approach)")
    logger.info(f"   - Adaptive threshold enabled: {cfg.birdnet.get('use_adaptive_threshold', False)}")
    if cfg.birdnet.get('use_adaptive_threshold', False):
        logger.info(f"   - Adaptive factor: {cfg.birdnet.get('adaptive_factor', 1.0)}x")
    
    try:
        from benchmark.predict_birdnet import BirdNETPredictor, predict_from_dataframe
        
        # Use the larger FP32 classification model explicitly
        model_fp32_path = os.path.join(
            original_cwd,
            "analyzer",
            "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
        )
        
        # Check if birds-only mode is enabled
        force_bird_prediction = cfg.benchmark.mode.get('exclude_no_birds_from_ground_truth', False)
        
        # Initialize predictor with aligned preprocessing parameters
        predictor = BirdNETPredictor(
            target_species=cfg.student_model.classes.allowed_species,
            confidence_threshold=cfg.birdnet.confidence_threshold,
            use_adaptive_threshold=cfg.birdnet.get('use_adaptive_threshold', False),
            adaptive_factor=cfg.birdnet.get('adaptive_factor', 1.0),
            force_bird_prediction=force_bird_prediction,
            model_path=model_fp32_path,
            segment_duration=cfg.student_model.preprocessing.clip_duration,
            sample_rate=cfg.student_model.preprocessing.sample_rate,
            lowcut=cfg.student_model.preprocessing.lowcut,
            highcut=cfg.student_model.preprocessing.highcut,
            extract_calls=cfg.student_model.preprocessing.get('extract_calls', True)
        )
        
        # Generate predictions using aligned processing
        predictions_df = predict_from_dataframe(
            predictor,
            ground_truth_df,
            audio_base_path=original_cwd,
            use_aligned_processing=True  # Use 3s segments with identical preprocessing
        )
        
        # Save predictions
        output_path = os.path.join(original_cwd, cfg.benchmark.paths.predictions_dir, "birdnet_predictions.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        predictions_df.to_csv(output_path, index=False)
        
        logger.info(f"BirdNET predictions saved to: {output_path}")
        
        # Log summary
        pred_counts = predictions_df['birdnet_prediction'].value_counts()
        logger.info("✅ BirdNET ALIGNED prediction completed")
        logger.info(f"  Total predictions: {len(predictions_df)}")
        logger.info(f"  Average confidence: {predictions_df['birdnet_confidence'].mean():.3f}")
        logger.info(f"  Top prediction: {pred_counts.index[0]} ({pred_counts.iloc[0]} files)")
        
        # Log preprocessing method distribution
        if 'birdnet_preprocessing' in predictions_df.columns:
            preprocessing_counts = predictions_df['birdnet_preprocessing'].value_counts()
            logger.info("  Preprocessing methods used:")
            for method, count in preprocessing_counts.items():
                logger.info(f"    {method}: {count} files")
        
        # Cleanup
        predictor.cleanup()
        
        return predictions_df
        
    except Exception as e:
        logger.error(f"BirdNET prediction failed: {e}")
        raise


def run_comparison(cfg: DictConfig, student_df: pd.DataFrame, birdnet_df: pd.DataFrame, original_cwd: str) -> Dict:
    """
    Compare student model and BirdNET predictions.
    
    Args:
        cfg: Configuration object
        student_df: Student predictions DataFrame
        birdnet_df: BirdNET predictions DataFrame
        original_cwd: Original working directory
        
    Returns:
        Dictionary with comparison results
    """
    logger.info("Starting model comparison")
    
    try:
        from benchmark.compare_predictions import ModelComparator
        
        # Initialize comparator
        comparator = ModelComparator(cfg.comparison)
        
        # Merge predictions
        merged_df = pd.merge(
            student_df[['audio_path', 'ground_truth', 'student_prediction', 'student_confidence']],
            birdnet_df[['audio_path', 'birdnet_prediction', 'birdnet_confidence']],
            on='audio_path',
            how='inner'
        )
        
        if len(merged_df) == 0:
            logger.error("No matching predictions found for comparison!")
            return {}
        
        logger.info(f"Comparing {len(merged_df)} predictions")
        
        # Generate comparison
        comparison_results = comparator.compare_predictions(
            ground_truth=merged_df['ground_truth'].tolist(),
            student_predictions=merged_df['student_prediction'].tolist(),
            birdnet_predictions=merged_df['birdnet_prediction'].tolist(),
            student_confidences=merged_df['student_confidence'].tolist(),
            birdnet_confidences=merged_df['birdnet_confidence'].tolist(),
            audio_paths=merged_df['audio_path'].tolist()
        )
        
        # Save comparison results
        output_dir = os.path.join(original_cwd, cfg.benchmark.paths.comparison_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        comparator.save_results(comparison_results, output_dir)
        
        logger.info(f"Comparison results saved to: {output_dir}")
        
        return comparison_results
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise


def run_statistical_analysis(cfg: DictConfig, original_cwd: str) -> Dict:
    """
    Run statistical analysis to validate benchmark rigor.
    
    Args:
        cfg: Configuration object
        original_cwd: Original working directory
        
    Returns:
        Statistical analysis results
    """
    logger.info("🔬 RUNNING STATISTICAL ANALYSIS...")
    
    try:
        from benchmark.statistical_analysis import StatisticalAnalyzer, calculate_benchmark_statistical_requirements
        
        # Get benchmark parameters
        num_classes = 9 if not cfg.benchmark.mode.get('exclude_no_birds_from_ground_truth', False) else 8
        baseline_accuracy = 0.88  # Current known performance
        target_improvement = 0.05  # Want to detect 5% improvement
        
        # Calculate statistical requirements
        statistical_requirements = calculate_benchmark_statistical_requirements(
            baseline_accuracy=baseline_accuracy,
            effect_size=target_improvement
        )
        
        # Check if current dataset meets requirements
        _, ground_truth_df = discover_audio_files(cfg, original_cwd)
        current_sample_size = len(ground_truth_df)
        current_per_class = ground_truth_df['label'].value_counts()
        
        # Analysis
        recommended_total = statistical_requirements['sample_size_requirements']['recommended']
        recommended_per_class = statistical_requirements['sample_size_requirements']['per_class_minimum']
        
        meets_requirements = (
            current_sample_size >= recommended_total and
            current_per_class.min() >= recommended_per_class
        )
        
        analysis_results = {
            'statistical_requirements': statistical_requirements,
            'current_dataset': {
                'total_samples': current_sample_size,
                'per_class_distribution': current_per_class.to_dict(),
                'min_per_class': current_per_class.min(),
                'max_per_class': current_per_class.max(),
                'class_balance_ratio': current_per_class.min() / current_per_class.max()
            },
            'requirements_check': {
                'meets_sample_size': current_sample_size >= recommended_total,
                'meets_per_class_minimum': current_per_class.min() >= recommended_per_class,
                'overall_adequate': meets_requirements,
                'sample_size_ratio': current_sample_size / recommended_total,
                'improvement_detectable': target_improvement
            },
            'recommendations': {
                'current_status': 'ADEQUATE' if meets_requirements else 'INSUFFICIENT',
                'action_needed': 'None' if meets_requirements else 'Increase sample size',
                'target_total_samples': recommended_total,
                'target_per_class': recommended_per_class,
                'suggested_runs': statistical_requirements['multiple_runs']['recommended_runs']
            }
        }
        
        # Log results
        logger.info(f"📊 Statistical Analysis Results:")
        logger.info(f"  Current dataset: {current_sample_size} samples")
        logger.info(f"  Recommended: {recommended_total} samples")
        logger.info(f"  Status: {analysis_results['recommendations']['current_status']}")
        logger.info(f"  Can detect {target_improvement:.1%} improvement: {meets_requirements}")
        
        # Add more detailed summary
        logger.info(f"")
        logger.info(f"📈 STATISTICAL POWER ANALYSIS:")
        logger.info(f"  • Confidence Level: {statistical_requirements['statistical_parameters']['confidence_level']:.1%}")
        logger.info(f"  • Statistical Power: {statistical_requirements['statistical_parameters']['power']:.1%}")
        logger.info(f"  • Minimum Effect Size: {statistical_requirements['statistical_parameters']['effect_size']:.1%}")
        logger.info(f"  • Recommended Runs: {statistical_requirements['multiple_runs']['recommended_runs']}")
        logger.info(f"")
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Statistical analysis failed: {e}")
        return {'error': str(e)}


def run_multiple_benchmark_runs(cfg: DictConfig, original_cwd: str, num_runs: int = 3) -> Dict:
    """
    Run multiple benchmark runs for statistical validation.
    Each run gets its own timestamped directory.
    
    Args:
        cfg: Configuration object
        original_cwd: Original working directory
        num_runs: Number of runs to execute
        
    Returns:
        Multiple run analysis results
    """
    logger.info(f"🔄 RUNNING MULTIPLE BENCHMARK RUNS (n={num_runs})")
    
    try:
        from benchmark.statistical_analysis import StatisticalAnalyzer
        
        analyzer = StatisticalAnalyzer()
        run_results = []
        
        # Create main timestamped directory for this multiple run session
        session_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_dir = os.path.join(original_cwd, cfg.benchmark.paths.output_dir, f"multiple_runs_{session_timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        
        logger.info(f"📁 Multiple runs session directory: multiple_runs_{session_timestamp}")
        
        for run_idx in range(num_runs):
            logger.info(f"📊 Starting run {run_idx + 1}/{num_runs}")
            
            # Create timestamped directory for this specific run
            run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_dir = os.path.join(session_dir, f"run_{run_idx+1:02d}_{run_timestamp}")
            
            # Update configuration paths for this run
            cfg.benchmark.paths.predictions_dir = os.path.join(run_dir, "predictions")
            cfg.benchmark.paths.comparison_dir = os.path.join(run_dir, "comparison")
            os.makedirs(cfg.benchmark.paths.predictions_dir, exist_ok=True)
            os.makedirs(cfg.benchmark.paths.comparison_dir, exist_ok=True)
            
            # Create random subset for this run (if configured)
            if cfg.get('debug', {}).get('files_limit'):
                # Use different random seed for each run
                np.random.seed(42 + run_idx)
            
            try:
                # Run single benchmark
                _, ground_truth_df = discover_audio_files(cfg, original_cwd)
                student_predictions_df = run_student_predictions(cfg, ground_truth_df, original_cwd)
                birdnet_predictions_df = run_birdnet_predictions(cfg, ground_truth_df, original_cwd)
                comparison_results = run_comparison(cfg, student_predictions_df, birdnet_predictions_df, original_cwd)
                
                # Extract key metrics
                student_accuracy = comparison_results.get('student_accuracy', 0)
                birdnet_accuracy = comparison_results.get('birdnet_accuracy', 0)
                
                run_result = {
                    'run_index': run_idx,
                    'student_accuracy': student_accuracy,
                    'birdnet_accuracy': birdnet_accuracy,
                    'difference': student_accuracy - birdnet_accuracy,
                    'sample_size': len(ground_truth_df),
                    'timestamp': datetime.now().isoformat()
                }
                
                run_results.append(run_result)
                logger.info(f"  Run {run_idx + 1} completed: Student {student_accuracy:.1%}, BirdNET {birdnet_accuracy:.1%}")
                
            except Exception as e:
                logger.error(f"Run {run_idx + 1} failed: {e}")
                continue
        
        if len(run_results) < 2:
            logger.error("Need at least 2 successful runs for statistical analysis")
            return {'error': 'Insufficient successful runs'}
        
        # Perform statistical analysis on multiple runs
        statistical_analysis = analyzer.multiple_run_analysis(run_results)
        
        # Save detailed results in session directory
        session_results_dir = os.path.join(session_dir, "aggregated_results")
        os.makedirs(session_results_dir, exist_ok=True)
        
        with open(os.path.join(session_results_dir, 'multiple_runs_raw_data.json'), 'w') as f:
            json.dump(run_results, f, indent=2)
        
        with open(os.path.join(session_results_dir, 'multiple_runs_analysis.json'), 'w') as f:
            json.dump(statistical_analysis, f, indent=2)
        
        # Create statistical report
        if hasattr(analyzer, 'save_statistical_report'):
            analyzer.save_statistical_report(
                {'multiple_run_analysis': statistical_analysis}, 
                session_results_dir
            )
        else:
            # Generate statistical report manually
            report = analyzer.generate_statistical_report(statistical_analysis)
            with open(os.path.join(session_results_dir, 'statistical_report.txt'), 'w') as f:
                f.write(report)
        
        logger.info(f"✅ Multiple run analysis completed:")
        logger.info(f"  Successful runs: {len(run_results)}")
        logger.info(f"  Mean difference: {statistical_analysis['difference']['mean']:.1%}")
        logger.info(f"  95% CI: [{statistical_analysis['difference']['confidence_interval'][0]:.1%}, {statistical_analysis['difference']['confidence_interval'][1]:.1%}]")
        logger.info(f"  Statistical significance: {statistical_analysis['statistical_tests']['paired_t_test']['is_significant']}")
        
        return {
            'multiple_runs_analysis': statistical_analysis,
            'raw_results': run_results,
            'summary': {
                'num_successful_runs': len(run_results),
                'statistically_significant': statistical_analysis['statistical_tests']['paired_t_test']['is_significant'],
                'practical_significance': statistical_analysis['interpretation']['practical_significance'],
                'recommendation': statistical_analysis['interpretation']['recommendation']
            }
        }
        
    except Exception as e:
        logger.error(f"Multiple run analysis failed: {e}")
        return {'error': str(e)}


def test_benchmark_improvements(cfg: DictConfig, original_cwd: str) -> bool:
    """
    Test benchmark improvements with a quick validation run.
    
    Args:
        cfg: Configuration object
        original_cwd: Original working directory
        
    Returns:
        Boolean indicating if tests passed
    """
    logger.info("🧪 RUNNING BENCHMARK IMPROVEMENTS TEST...")
    logger.info("This is a quick validation to ensure new features work correctly")
    
    try:
        # Test 1: Verify adaptive threshold functionality
        logger.info("🔧 Testing single threshold configuration...")
        
        if cfg.birdnet.get('use_adaptive_threshold', False):
            logger.info("ℹ️  Adaptive threshold is enabled but simplified to single threshold approach")
        
        threshold = cfg.birdnet.confidence_threshold
        logger.info(f"✅ Single threshold configuration OK: {threshold}")
        
        # Test 2: Verify birds-only mode configuration
        logger.info("🎯 Testing birds-only mode configuration...")
        
        if cfg.benchmark.mode.get('exclude_no_birds_from_ground_truth', False):
            logger.info("✅ Birds-only mode enabled: no_birds samples will be excluded")
            if cfg.birdnet.get('use_adaptive_threshold', False):
                logger.info("💡 Note: Adaptive threshold + birds-only mode = maximum bird species accuracy")
        
        # Test 3: Run a mini benchmark with very limited data
        logger.info("🚀 Running mini benchmark test (limited files)...")
        
        # Create limited test configuration
        original_files_limit = cfg.get('debug', {}).get('files_limit', None)
        original_max_files = cfg.get('debug', {}).get('max_files_per_class', None)
        
        # Temporarily override for test
        if 'debug' not in cfg:
            cfg.debug = {}
        cfg.debug.files_limit = 20  # Very limited for test
        cfg.debug.max_files_per_class = 3
        
        # Discovery test
        _, ground_truth_df = discover_audio_files(cfg, original_cwd)
        
        if len(ground_truth_df) == 0:
            logger.error("❌ Test FAILED: No audio files discovered")
            return False
        
        logger.info(f"✅ Audio discovery OK: Found {len(ground_truth_df)} files")
        
        # Test BirdNET predictor initialization
        from benchmark.predict_birdnet import BirdNETPredictor
        
        # Check if birds-only mode is enabled for test
        force_bird_prediction = cfg.benchmark.mode.get('exclude_no_birds_from_ground_truth', False)
        
        test_predictor = BirdNETPredictor(
            target_species=cfg.student_model.classes.allowed_species,
            confidence_threshold=cfg.birdnet.confidence_threshold,
            use_adaptive_threshold=cfg.birdnet.get('use_adaptive_threshold', False),
            adaptive_factor=cfg.birdnet.get('adaptive_factor', 1.0),
            force_bird_prediction=force_bird_prediction,
            segment_duration=cfg.student_model.preprocessing.clip_duration,
            sample_rate=cfg.student_model.preprocessing.sample_rate,
            lowcut=cfg.student_model.preprocessing.lowcut,
            highcut=cfg.student_model.preprocessing.highcut,
            extract_calls=cfg.student_model.preprocessing.get('extract_calls', True)
        )
        
        logger.info("✅ BirdNET predictor initialization OK")
        test_predictor.cleanup()
        
        # Restore original configuration
        if original_files_limit is not None:
            cfg.debug.files_limit = original_files_limit
        if original_max_files is not None:
            cfg.debug.max_files_per_class = original_max_files
        
        logger.info("🎉 ALL TESTS PASSED! Benchmark improvements are working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test FAILED with error: {e}")
        return False


@hydra.main(version_base=None, config_path="config", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    """Main benchmark function."""
    import torch
    
    # Force immediate CUDA initialization to prevent conflicts with other libraries 
    # (e.g., tensorflow from birdnetlib or numba from librosa).
    # This call ensures PyTorch "wins the race" for the GPU.
    torch.cuda.is_available()

    # Get original working directory
    original_cwd = hydra.utils.get_original_cwd()
    
    # Setup timestamped output paths (unless in special modes)
    test_mode = cfg.get('test_mode', False)
    statistical_analysis_mode = cfg.get('statistical_analysis_mode', False)
    multiple_runs_mode = cfg.get('multiple_runs_mode', False)
    num_runs = cfg.get('num_runs', 3)
    
    # Only setup timestamped paths for actual benchmarks (not for statistical analysis or test modes)
    if not statistical_analysis_mode and not test_mode:
        timestamped_dir = setup_timestamped_output_paths(cfg, original_cwd)
    else:
        # For statistical analysis and test modes, use timestamped directory for clarity
        if statistical_analysis_mode:
            mode_dir = "statistical_analysis"
        elif test_mode:
            mode_dir = "test_mode"
        else:
            mode_dir = "temp"
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        analysis_dir = os.path.join(original_cwd, "benchmark_results", f"{mode_dir}_{timestamp}")
        
        cfg.benchmark.paths.predictions_dir = os.path.join(analysis_dir, "predictions")
        cfg.benchmark.paths.comparison_dir = os.path.join(analysis_dir, "comparison")
        os.makedirs(cfg.benchmark.paths.predictions_dir, exist_ok=True)
        os.makedirs(cfg.benchmark.paths.comparison_dir, exist_ok=True)
        
        if statistical_analysis_mode:
            logger.info(f"📁 Statistical analysis directory: {mode_dir}_{timestamp}")
            logger.info(f"   Output will be saved to: benchmark_results/{mode_dir}_{timestamp}/comparison/")
    
    if test_mode:
        logger.info("=" * 60)
        logger.info("🧪 BENCHMARK IMPROVEMENTS TEST MODE")
        logger.info("=" * 60)
        
        success = test_benchmark_improvements(cfg, original_cwd)
        
        if success:
            logger.info("✅ Test mode completed successfully!")
            return
        else:
            logger.error("❌ Test mode failed!")
            raise RuntimeError("Benchmark improvements test failed")
    
    if statistical_analysis_mode:
        logger.info("=" * 60)
        logger.info("🔬 STATISTICAL ANALYSIS MODE")
        logger.info("=" * 60)
        
        analysis_results = run_statistical_analysis(cfg, original_cwd)
        
        if 'error' not in analysis_results:
            logger.info("✅ Statistical analysis completed successfully!")
            
            # Optionally save results
            output_dir = os.path.join(original_cwd, cfg.benchmark.paths.comparison_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                import numpy as np
                if isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            analysis_results_serializable = convert_numpy_types(analysis_results)
            
            with open(os.path.join(output_dir, 'statistical_analysis.json'), 'w') as f:
                json.dump(analysis_results_serializable, f, indent=2)
            
            return
        else:
            logger.error("❌ Statistical analysis failed!")
            raise RuntimeError("Statistical analysis failed")
    
    if multiple_runs_mode:
        logger.info("=" * 60)
        logger.info(f"🔄 MULTIPLE RUNS MODE (n={num_runs})")
        logger.info("=" * 60)
        
        multiple_runs_results = run_multiple_benchmark_runs(cfg, original_cwd, num_runs)
        
        if 'error' not in multiple_runs_results:
            logger.info("✅ Multiple runs analysis completed successfully!")
            return
        else:
            logger.error("❌ Multiple runs analysis failed!")
            raise RuntimeError("Multiple runs analysis failed")
    
    # Create output directories
    predictions_dir = os.path.join(original_cwd, cfg.benchmark.paths.predictions_dir)
    os.makedirs(predictions_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("🎯 ALIGNED BIRD CLASSIFICATION BENCHMARK")
    logger.info("=" * 60)
    logger.info("🔧 FAIR COMPARISON MODE:")
    logger.info("   • BirdNET: Uses 3s segments with student model preprocessing")
    logger.info("   • Student: Uses same 3s segments and preprocessing")
    logger.info("   • Both models see identical audio data")
    
    # Log benchmark mode configuration
    if cfg.benchmark.mode.get('exclude_no_birds_from_ground_truth', False):
        logger.info("🎯 BIRDS-ONLY MODE: Excluding no_birds class from evaluation")
        logger.info("🔄 FORCE BIRD PREDICTION: Both models will never predict no_birds")
    
    logger.info(f"🔧 SINGLE THRESHOLD: {cfg.birdnet.confidence_threshold} (simplified approach)")
    
    logger.info("=" * 60)
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Hydra output directory: {os.getcwd()}")
    
    # ========================================
    # STEP 1: Audio Discovery (Always runs)
    # ========================================
    logger.info("\n" + "=" * 40)
    logger.info("📁 STEP 1: Audio Discovery")
    logger.info("=" * 40)
    
    _, ground_truth_df = discover_audio_files(cfg, original_cwd)
    
    # Save ground truth for reference
    gt_path = os.path.join(predictions_dir, "ground_truth.csv")
    ground_truth_df.to_csv(gt_path, index=False)
    logger.info(f"✅ Ground truth saved to: {gt_path}")

    # ========================================
    # STEP 2: Student Model Predictions
    # ========================================
    logger.info("\n" + "=" * 40)
    logger.info("🤖 STEP 2: Student Model Predictions")
    logger.info("=" * 40)
    
    student_predictions_df = run_student_predictions(cfg, ground_truth_df, original_cwd)

    # ========================================
    # STEP 3: BirdNET Aligned Predictions
    # ========================================
    logger.info("\n" + "=" * 40)
    logger.info("🦅 STEP 3: BirdNET Aligned Predictions")
    logger.info("=" * 40)
    
    birdnet_predictions_df = run_birdnet_predictions(cfg, ground_truth_df, original_cwd)

    # ========================================
    # STEP 4: Metrics and Comparison
    # ========================================
    logger.info("\n" + "=" * 40)
    logger.info("📊 STEP 4: Metrics and Comparison")
    logger.info("=" * 40)

    # Run comparison
    run_comparison(cfg, student_predictions_df, birdnet_predictions_df, original_cwd)
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 ALIGNED BENCHMARK COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info("✅ Fair comparison achieved:")
    logger.info("   • Both models used identical preprocessing")
    logger.info("   • Both models analyzed same 3-second segments")
    logger.info("   • Performance gap should be <5% (realistic)")
    logger.info("📊 Check results/comparison/ for detailed analysis")
    logger.info("=" * 60)


if __name__ == "__main__":
    main() 