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

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)


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
    
    # Process no_birds dataset
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
        
        # Initialize predictor
        predictor = StudentModelPredictor(
            model_path=os.path.join(original_cwd, cfg.benchmark.paths.student_model),
            config_path=os.path.join(original_cwd, cfg.benchmark.paths.student_config),
            device=cfg.student_model.inference.device,
            confidence_threshold=cfg.student_model.inference.confidence_threshold
        )
        
        # Generate predictions
        predictions_df = predict_from_dataframe(
            predictor,
            ground_truth_df,
            audio_base_path=original_cwd
        )
        
        # Save predictions
        benchmark_dir = os.path.join(original_cwd, "benchmark")
        output_path = os.path.join(benchmark_dir, cfg.benchmark.paths.predictions_dir, "student_predictions.csv")
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
    Run BirdNET predictions.
    
    Args:
        cfg: Configuration object
        ground_truth_df: Ground truth DataFrame
        original_cwd: Original working directory
        
    Returns:
        DataFrame with BirdNET predictions
    """
    logger.info("Starting BirdNET predictions")
    
    try:
        from benchmark.predict_birdnet import BirdNETPredictor, predict_from_dataframe
        
        # Initialize predictor
        predictor = BirdNETPredictor(
            target_species=cfg.student_model.classes.allowed_species,
            confidence_threshold=cfg.birdnet.confidence_threshold
        )
        
        # Generate predictions
        predictions_df = predict_from_dataframe(
            predictor,
            ground_truth_df,
            audio_base_path=original_cwd
        )
        
        # Save predictions
        benchmark_dir = os.path.join(original_cwd, "benchmark")
        output_path = os.path.join(benchmark_dir, cfg.benchmark.paths.predictions_dir, "birdnet_predictions.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        predictions_df.to_csv(output_path, index=False)
        
        logger.info(f"BirdNET predictions saved to: {output_path}")
        
        # Log summary
        pred_counts = predictions_df['birdnet_prediction'].value_counts()
        logger.info("✅ BirdNET prediction completed")
        logger.info(f"  Total predictions: {len(predictions_df)}")
        logger.info(f"  Average confidence: {predictions_df['birdnet_confidence'].mean():.3f}")
        logger.info(f"  Top prediction: {pred_counts.index[0]} ({pred_counts.iloc[0]} files)")
        
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
        benchmark_dir = os.path.join(original_cwd, "benchmark")
        output_dir = os.path.join(benchmark_dir, cfg.benchmark.paths.comparison_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        comparator.save_results(comparison_results, output_dir)
        
        logger.info(f"Comparison results saved to: {output_dir}")
        
        return comparison_results
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise


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
    
    # Create output directories
    benchmark_dir = os.path.join(original_cwd, "benchmark")
    predictions_dir = os.path.join(benchmark_dir, cfg.benchmark.paths.predictions_dir)
    os.makedirs(predictions_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("🐦 BIRD CLASSIFICATION BENCHMARK")
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
    # STEP 3: BirdNET Predictions
    # ========================================
    logger.info("\n" + "=" * 40)
    logger.info("🦉 STEP 3: BirdNET Predictions")
    logger.info("=" * 40)
    
    birdnet_predictions_df = run_birdnet_predictions(cfg, ground_truth_df, original_cwd)

    # ========================================
    # STEP 4: Metrics and Comparison
    # ========================================
    logger.info("\n" + "=" * 40)
    logger.info("📊 STEP 4: Metrics and Comparison")
    logger.info("=" * 40)

    # Merge predictions
    student_preds = student_predictions_df[['audio_path', 'student_prediction', 'student_confidence']]
    birdnet_preds = birdnet_predictions_df[['audio_path', 'birdnet_prediction', 'birdnet_confidence']]
    combined_df = pd.merge(ground_truth_df, student_preds, on='audio_path')
    combined_df = pd.merge(combined_df, birdnet_preds, on='audio_path')

    # Run comparison
    run_comparison(cfg, combined_df, original_cwd)
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 BENCHMARK COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main() 