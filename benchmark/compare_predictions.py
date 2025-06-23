#!/usr/bin/env python3
"""
Model Comparison Script

Compares predictions from student model and BirdNET against ground truth.
Generates comprehensive evaluation metrics and analysis.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, confusion_matrix
)
import hydra
from omegaconf import DictConfig, ListConfig

logger = logging.getLogger(__name__)

def convert_listconfig_to_list(obj):
    """
    Recursively convert ListConfig objects to regular lists for JSON serialization.
    """
    if isinstance(obj, ListConfig):
        return [convert_listconfig_to_list(item) for item in obj]
    elif isinstance(obj, DictConfig):
        return {key: convert_listconfig_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, dict):
        return {key: convert_listconfig_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_listconfig_to_list(item) for item in obj]
    else:
        return obj

class ModelComparator:
    """
    Compares predictions from student model and BirdNET against ground truth.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize model comparator.
        
        Args:
            config: Configuration for comparison
        """
        self.config = config
        # Extract exclude_classes configuration
        self.exclude_classes = getattr(config.metrics, 'exclude_classes', [])
        self.birds_only_mode = getattr(config.get('benchmark', {}).get('mode', {}), 'birds_only', False)
        
        logger.info(f"ModelComparator initialized with exclude_classes: {self.exclude_classes}")
        logger.info(f"Birds-only mode: {self.birds_only_mode}")
        
    def compare_predictions(self, 
                          ground_truth: List[str],
                          student_predictions: List[str], 
                          birdnet_predictions: List[str],
                          student_confidences: List[float],
                          birdnet_confidences: List[float],
                          audio_paths: List[str]) -> Dict:
        """
        Compare predictions from both models against ground truth.
        
        Args:
            ground_truth: List of true labels
            student_predictions: List of student model predictions
            birdnet_predictions: List of BirdNET predictions
            student_confidences: List of student model confidences
            birdnet_confidences: List of BirdNET confidences
            audio_paths: List of audio file paths
            
        Returns:
            Dictionary with comparison results
        """
        logger.info("Comparing model predictions")
        
        # Apply filtering for birds-only mode or exclude_classes
        if self.exclude_classes or self.birds_only_mode:
            logger.info(f"Applying class filtering - exclude_classes: {self.exclude_classes}, birds_only: {self.birds_only_mode}")
            (ground_truth, student_predictions, birdnet_predictions, 
             student_confidences, birdnet_confidences, audio_paths) = self._filter_excluded_classes(
                ground_truth, student_predictions, birdnet_predictions,
                student_confidences, birdnet_confidences, audio_paths
            )
            logger.info(f"After filtering: {len(ground_truth)} samples remaining")
        
        # Convert to numpy arrays for easier processing
        y_true = np.array(ground_truth)
        y_pred_student = np.array(student_predictions)
        y_pred_birdnet = np.array(birdnet_predictions)
        
        # Get unique classes (after filtering)
        all_classes = sorted(list(set(ground_truth + student_predictions + birdnet_predictions)))
        logger.info(f"Classes in analysis: {all_classes}")
        
        # Calculate metrics for both models
        student_metrics = self._calculate_metrics(y_true, y_pred_student, all_classes)
        birdnet_metrics = self._calculate_metrics(y_true, y_pred_birdnet, all_classes)
        
        # Agreement analysis
        agreement_analysis = self._analyze_agreement(
            y_true, y_pred_student, y_pred_birdnet,
            student_confidences, birdnet_confidences, audio_paths
        )
        
        # Per-class analysis
        per_class_analysis = self._per_class_analysis(y_true, y_pred_student, y_pred_birdnet, all_classes)
        
        # Confusion matrices
        confusion_matrices = self._generate_confusion_matrices(y_true, y_pred_student, y_pred_birdnet, all_classes)
        
        results = {
            'metrics': {
                'student_accuracy': student_metrics['accuracy'],
                'student_precision': student_metrics['precision_macro'],
                'student_recall': student_metrics['recall_macro'],
                'student_f1': student_metrics['f1_macro'],
                'birdnet_accuracy': birdnet_metrics['accuracy'],
                'birdnet_precision': birdnet_metrics['precision_macro'],
                'birdnet_recall': birdnet_metrics['recall_macro'],
                'birdnet_f1': birdnet_metrics['f1_macro']
            },
            'detailed_metrics': {
                'student': student_metrics,
                'birdnet': birdnet_metrics
            },
            'agreement_analysis': agreement_analysis,
            'per_class_analysis': per_class_analysis,
            'confusion_matrices': confusion_matrices,
            'summary': {
                'total_samples': len(ground_truth),
                'num_classes': len(all_classes),
                'classes': all_classes,
                'excluded_classes': self.exclude_classes,
                'birds_only_mode': self.birds_only_mode
            }
        }
        
        return results
    
    def _filter_excluded_classes(self, ground_truth: List[str], student_predictions: List[str], 
                                birdnet_predictions: List[str], student_confidences: List[float],
                                birdnet_confidences: List[float], audio_paths: List[str]) -> tuple:
        """
        Filter out excluded classes from all data.
        
        Args:
            ground_truth: List of true labels
            student_predictions: List of student model predictions
            birdnet_predictions: List of BirdNET predictions
            student_confidences: List of student model confidences
            birdnet_confidences: List of BirdNET confidences
            audio_paths: List of audio file paths
            
        Returns:
            Tuple of filtered lists
        """
        # Determine classes to exclude
        classes_to_exclude = set(self.exclude_classes)
        
        # In birds-only mode, automatically exclude no_birds
        if self.birds_only_mode:
            classes_to_exclude.add('no_birds')
        
        if not classes_to_exclude:
            return (ground_truth, student_predictions, birdnet_predictions,
                   student_confidences, birdnet_confidences, audio_paths)
        
        logger.info(f"Filtering out classes: {classes_to_exclude}")
        
        # Create mask for samples to keep
        keep_mask = []
        for i, gt in enumerate(ground_truth):
            # Keep sample if ground truth is NOT in excluded classes
            keep_sample = gt not in classes_to_exclude
            keep_mask.append(keep_sample)
        
        # Apply filtering
        filtered_ground_truth = [ground_truth[i] for i in range(len(ground_truth)) if keep_mask[i]]
        filtered_student_pred = [student_predictions[i] for i in range(len(student_predictions)) if keep_mask[i]]
        filtered_birdnet_pred = [birdnet_predictions[i] for i in range(len(birdnet_predictions)) if keep_mask[i]]
        filtered_student_conf = [student_confidences[i] for i in range(len(student_confidences)) if keep_mask[i]]
        filtered_birdnet_conf = [birdnet_confidences[i] for i in range(len(birdnet_confidences)) if keep_mask[i]]
        filtered_audio_paths = [audio_paths[i] for i in range(len(audio_paths)) if keep_mask[i]]
        
        # Now also filter predictions that are in excluded classes (convert them to most likely bird class)
        filtered_student_pred = self._remap_excluded_predictions(filtered_student_pred, classes_to_exclude)
        filtered_birdnet_pred = self._remap_excluded_predictions(filtered_birdnet_pred, classes_to_exclude)
        
        logger.info(f"Samples before filtering: {len(ground_truth)}")
        logger.info(f"Samples after filtering: {len(filtered_ground_truth)}")
        
        return (filtered_ground_truth, filtered_student_pred, filtered_birdnet_pred,
                filtered_student_conf, filtered_birdnet_conf, filtered_audio_paths)
    
    def _remap_excluded_predictions(self, predictions: List[str], excluded_classes: set) -> List[str]:
        """
        Remap predictions that are in excluded classes to 'unknown' or handle them appropriately.
        
        Args:
            predictions: List of predictions
            excluded_classes: Set of classes to exclude
            
        Returns:
            List of remapped predictions
        """
        # For birds-only mode, if a model predicts no_birds, it's essentially an error
        # We'll keep the prediction as-is for accuracy calculation
        # (this will be counted as incorrect if ground truth is a bird species)
        
        remapped = []
        for pred in predictions:
            if pred in excluded_classes:
                # In birds-only mode, keep the excluded prediction as-is
                # This will show up as an error in the confusion matrix
                remapped.append(pred)
            else:
                remapped.append(pred)
        
        return remapped
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]) -> Dict:
        """Calculate classification metrics."""
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
                'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
                'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
                'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
                'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
                'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
                'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            
            # Per-class metrics
            report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
            metrics['per_class'] = report
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return self._empty_metrics()
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary."""
        return {
            'accuracy': 0.0,
            'precision_micro': 0.0,
            'precision_macro': 0.0,
            'precision_weighted': 0.0,
            'recall_micro': 0.0,
            'recall_macro': 0.0,
            'recall_weighted': 0.0,
            'f1_micro': 0.0,
            'f1_macro': 0.0,
            'f1_weighted': 0.0,
            'per_class': {}
        }
    
    def _analyze_agreement(self, y_true: np.ndarray, y_pred_student: np.ndarray, y_pred_birdnet: np.ndarray,
                          student_confidences: List[float], birdnet_confidences: List[float], 
                          audio_paths: List[str]) -> Dict:
        """Analyze agreement between models."""
        
        student_correct = (y_true == y_pred_student)
        birdnet_correct = (y_true == y_pred_birdnet)
        
        both_correct = student_correct & birdnet_correct
        student_only = student_correct & (~birdnet_correct)
        birdnet_only = (~student_correct) & birdnet_correct
        both_incorrect = (~student_correct) & (~birdnet_correct)
        
        total = len(y_true)
        
        agreement = {
            'counts': {
                'both_correct': int(both_correct.sum()),
                'student_only_correct': int(student_only.sum()),
                'birdnet_only_correct': int(birdnet_only.sum()),
                'both_incorrect': int(both_incorrect.sum())
            },
            'percentages': {
                'both_correct': float(both_correct.sum() / total * 100),
                'student_only_correct': float(student_only.sum() / total * 100),
                'birdnet_only_correct': float(birdnet_only.sum() / total * 100),
                'both_incorrect': float(both_incorrect.sum() / total * 100)
            },
            'detailed_cases': []
        }
        
        # Store detailed cases for analysis
        for i in range(len(y_true)):
            if student_only[i] or birdnet_only[i] or both_incorrect[i]:  # Store interesting cases
                agreement['detailed_cases'].append({
                    'audio_path': audio_paths[i],
                    'true_label': str(y_true[i]),
                    'student_prediction': str(y_pred_student[i]),
                    'student_confidence': float(student_confidences[i]),
                    'birdnet_prediction': str(y_pred_birdnet[i]),
                    'birdnet_confidence': float(birdnet_confidences[i]),
                    'student_correct': bool(student_correct[i]),
                    'birdnet_correct': bool(birdnet_correct[i]),
                    'category': ('both_correct' if both_correct[i] else
                               'student_only_correct' if student_only[i] else
                               'birdnet_only_correct' if birdnet_only[i] else
                               'both_incorrect')
                })
        
        return agreement
    
    def _per_class_analysis(self, y_true: np.ndarray, y_pred_student: np.ndarray, 
                           y_pred_birdnet: np.ndarray, classes: List[str]) -> Dict:
        """Analyze per-class performance."""
        per_class = {}
        
        for class_name in classes:
            class_mask = (y_true == class_name)
            if class_mask.sum() == 0:
                continue
                
            class_true = y_true[class_mask]
            class_pred_student = y_pred_student[class_mask]
            class_pred_birdnet = y_pred_birdnet[class_mask]
            
            per_class[class_name] = {
                'support': int(class_mask.sum()),
                'student_accuracy': float((class_true == class_pred_student).mean()),
                'birdnet_accuracy': float((class_true == class_pred_birdnet).mean()),
                'student_predictions': int((y_pred_student == class_name).sum()),
                'birdnet_predictions': int((y_pred_birdnet == class_name).sum())
            }
        
        return per_class
    
    def _generate_confusion_matrices(self, y_true: np.ndarray, y_pred_student: np.ndarray,
                                   y_pred_birdnet: np.ndarray, classes: List[str]) -> Dict:
        """Generate confusion matrices."""
        try:
            student_cm = confusion_matrix(y_true, y_pred_student, labels=classes)
            birdnet_cm = confusion_matrix(y_true, y_pred_birdnet, labels=classes)
            
            return {
                'student': student_cm.tolist(),
                'birdnet': birdnet_cm.tolist(),
                'labels': classes
            }
        except Exception as e:
            logger.error(f"Error generating confusion matrices: {e}")
            n_classes = len(classes)
            return {
                'student': [[0] * n_classes for _ in range(n_classes)],
                'birdnet': [[0] * n_classes for _ in range(n_classes)],
                'labels': classes
            }
    
    def save_results(self, results: Dict, output_dir: str):
        """Save all comparison results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get custom naming from config if available
        naming_config = getattr(self.config, 'output', {}).get('naming', {})
        plots_config = getattr(self.config, 'output', {}).get('plots', {})
        
        # 1. JSON report (machine-readable)
        json_filename = naming_config.get('comparison_report', 'comparison_report.json')
        json_path = output_path / json_filename
        with open(json_path, 'w') as f:
            json.dump(convert_listconfig_to_list(results), f, indent=2)
        logger.info(f"JSON report saved to: {json_path}")
        
        # 2. Text summary (human-readable)
        summary_filename = naming_config.get('comparison_summary', 'comparison_summary.txt')
        text_path = output_path / summary_filename
        self._save_text_summary(results, text_path)
        
        # 3. Detailed cases CSV
        if results['agreement_analysis']['detailed_cases']:
            cases_df = pd.DataFrame(results['agreement_analysis']['detailed_cases'])
            cases_filename = naming_config.get('detailed_cases', 'detailed_cases.csv')
            cases_path = output_path / cases_filename
            cases_df.to_csv(cases_path, index=False)
            logger.info(f"Detailed cases saved to: {cases_path}")
        
        # 4. Comprehensive metrics comparison table
        self._save_metrics_comparison_table(results, output_path, naming_config)
        
        # 5. Per-class detailed metrics table
        self._save_per_class_metrics_table(results, output_path, naming_config)
        
        # 6. Visualizations
        self._save_plots(results, output_path, plots_config)
        
        # 7. Enhanced metrics visualization
        self._plot_comprehensive_metrics_comparison(results, output_path, plots_config)

    def _save_text_summary(self, results: Dict, file_path: Path):
        """Save human-readable text summary."""
        with open(file_path, 'w') as f:
            f.write("BIRD CLASSIFICATION MODEL COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary
            summary = results['summary']
            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total samples: {summary['total_samples']}\n")
            f.write(f"Number of classes: {summary['num_classes']}\n")
            f.write(f"Classes: {', '.join(summary['classes'])}\n\n")
            
            # Overall metrics
            metrics = results['metrics']
            f.write("OVERALL METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Student Model:\n")
            f.write(f"  Accuracy: {metrics['student_accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['student_precision']:.4f}\n")
            f.write(f"  Recall: {metrics['student_recall']:.4f}\n")
            f.write(f"  F1-Score: {metrics['student_f1']:.4f}\n\n")
            
            f.write(f"BirdNET:\n")
            f.write(f"  Accuracy: {metrics['birdnet_accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['birdnet_precision']:.4f}\n")
            f.write(f"  Recall: {metrics['birdnet_recall']:.4f}\n")
            f.write(f"  F1-Score: {metrics['birdnet_f1']:.4f}\n\n")
            
            # Agreement analysis
            agreement = results['agreement_analysis']
            f.write("AGREEMENT ANALYSIS\n")
            f.write("-" * 20 + "\n")
            counts = agreement['counts']
            percentages = agreement['percentages']
            f.write(f"Both correct: {counts['both_correct']} ({percentages['both_correct']:.1f}%)\n")
            f.write(f"Student only correct: {counts['student_only_correct']} ({percentages['student_only_correct']:.1f}%)\n")
            f.write(f"BirdNET only correct: {counts['birdnet_only_correct']} ({percentages['birdnet_only_correct']:.1f}%)\n")
            f.write(f"Both incorrect: {counts['both_incorrect']} ({percentages['both_incorrect']:.1f}%)\n\n")
            
            # Per-class analysis
            f.write("PER-CLASS ANALYSIS\n")
            f.write("-" * 20 + "\n")
            for class_name, class_data in results['per_class_analysis'].items():
                f.write(f"{class_name} (support: {class_data['support']}):\n")
                f.write(f"  Student accuracy: {class_data['student_accuracy']:.4f}\n")
                f.write(f"  BirdNET accuracy: {class_data['birdnet_accuracy']:.4f}\n")
                f.write(f"  Student predictions: {class_data['student_predictions']}\n")
                f.write(f"  BirdNET predictions: {class_data['birdnet_predictions']}\n\n")
        
        logger.info(f"Text summary saved to: {file_path}")
    
    def _save_plots(self, results: Dict, output_dir: Path, plots_config: Dict = None):
        """Generate and save visualization plots."""
        try:
            # Set style
            plt.style.use('default')
            
            if plots_config is None:
                plots_config = {}
            
            # 1. Confusion matrices
            self._plot_confusion_matrices(results, output_dir, plots_config)
            
            # 2. Agreement analysis
            self._plot_agreement_analysis(results, output_dir, plots_config)
            
            # 3. Per-class comparison
            self._plot_per_class_comparison(results, output_dir, plots_config)
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
    
    def _plot_confusion_matrices(self, results: Dict, output_dir: Path, plots_config: Dict = None):
        """Plot confusion matrices for both models."""
        try:
            cm_data = results['confusion_matrices']
            labels = cm_data['labels']
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Student model
            student_cm = np.array(cm_data['student'])
            sns.heatmap(
                student_cm, 
                annot=True, 
                fmt='d', 
                xticklabels=labels,
                yticklabels=labels,
                ax=axes[0],
                cmap='Blues'
            )
            axes[0].set_title('Student Model Confusion Matrix')
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('True')
            
            # BirdNET
            birdnet_cm = np.array(cm_data['birdnet'])
            sns.heatmap(
                birdnet_cm,
                annot=True,
                fmt='d',
                xticklabels=labels,
                yticklabels=labels,
                ax=axes[1],
                cmap='Blues'
            )
            axes[1].set_title('BirdNET Confusion Matrix')
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('True')
            
            plt.tight_layout()
            filename = plots_config.get('confusion_matrices', 'confusion_matrices.png') if plots_config else 'confusion_matrices.png'
            plot_path = output_dir / filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrices saved to: {plot_path}")
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrices: {e}")
    
    def _plot_agreement_analysis(self, results: Dict, output_dir: Path, plots_config: Dict = None):
        """Plot agreement analysis."""
        try:
            agreement = results['agreement_analysis']
            counts = agreement['counts']
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Pie chart
            labels = ['Both Correct', 'Student Only', 'BirdNET Only', 'Both Incorrect']
            values = [
                counts['both_correct'],
                counts['student_only_correct'],
                counts['birdnet_only_correct'],
                counts['both_incorrect']
            ]
            colors = ['green', 'blue', 'orange', 'red']
            
            axes[0].pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[0].set_title('Model Agreement Analysis')
            
            # Bar chart
            categories = ['Both\nCorrect', 'Student\nOnly', 'BirdNET\nOnly', 'Both\nIncorrect']
            axes[1].bar(categories, values, color=colors, alpha=0.7)
            axes[1].set_title('Agreement Analysis (Counts)')
            axes[1].set_ylabel('Number of Samples')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = plots_config.get('agreement_analysis', 'agreement_analysis.png') if plots_config else 'agreement_analysis.png'
            plot_path = output_dir / filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Agreement analysis saved to: {plot_path}")
            
        except Exception as e:
            logger.error(f"Error plotting agreement analysis: {e}")
    
    def _plot_per_class_comparison(self, results: Dict, output_dir: Path, plots_config: Dict = None):
        """Plot per-class accuracy comparison."""
        try:
            per_class = results['per_class_analysis']
            
            if not per_class:
                return
            
            classes = list(per_class.keys())
            student_acc = [per_class[c]['student_accuracy'] for c in classes]
            birdnet_acc = [per_class[c]['birdnet_accuracy'] for c in classes]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(classes))
            width = 0.35
            
            ax.bar(x - width/2, student_acc, width, label='Student Model', alpha=0.8, color='blue')
            ax.bar(x + width/2, birdnet_acc, width, label='BirdNET', alpha=0.8, color='orange')
            
            ax.set_xlabel('Classes')
            ax.set_ylabel('Accuracy')
            ax.set_title('Per-Class Accuracy Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(classes, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.0)
            
            plt.tight_layout()
            filename = plots_config.get('per_class_accuracy', 'per_class_accuracy.png') if plots_config else 'per_class_accuracy.png'
            plot_path = output_dir / filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Per-class comparison saved to: {plot_path}")
            
        except Exception as e:
            logger.error(f"Error plotting per-class comparison: {e}")

    def _save_metrics_comparison_table(self, results: Dict, output_dir: Path, naming_config: Dict = None):
        """Save comprehensive metrics comparison table."""
        try:
            student_metrics = results['detailed_metrics']['student']
            birdnet_metrics = results['detailed_metrics']['birdnet']
            
            # Create comparison table
            comparison_data = {
                'Metric': [
                    'Accuracy',
                    'Precision (Micro)',
                    'Precision (Macro)', 
                    'Precision (Weighted)',
                    'Recall (Micro)',
                    'Recall (Macro)',
                    'Recall (Weighted)',
                    'F1-Score (Micro)',
                    'F1-Score (Macro)',
                    'F1-Score (Weighted)'
                ],
                'Student Model': [
                    f"{student_metrics['accuracy']:.4f}",
                    f"{student_metrics['precision_micro']:.4f}",
                    f"{student_metrics['precision_macro']:.4f}",
                    f"{student_metrics['precision_weighted']:.4f}",
                    f"{student_metrics['recall_micro']:.4f}",
                    f"{student_metrics['recall_macro']:.4f}",
                    f"{student_metrics['recall_weighted']:.4f}",
                    f"{student_metrics['f1_micro']:.4f}",
                    f"{student_metrics['f1_macro']:.4f}",
                    f"{student_metrics['f1_weighted']:.4f}"
                ],
                'BirdNET': [
                    f"{birdnet_metrics['accuracy']:.4f}",
                    f"{birdnet_metrics['precision_micro']:.4f}",
                    f"{birdnet_metrics['precision_macro']:.4f}",
                    f"{birdnet_metrics['precision_weighted']:.4f}",
                    f"{birdnet_metrics['recall_micro']:.4f}",
                    f"{birdnet_metrics['recall_macro']:.4f}",
                    f"{birdnet_metrics['recall_weighted']:.4f}",
                    f"{birdnet_metrics['f1_micro']:.4f}",
                    f"{birdnet_metrics['f1_macro']:.4f}",
                    f"{birdnet_metrics['f1_weighted']:.4f}"
                ]
            }
            
            # Calculate differences
            differences = []
            for i, metric_name in enumerate(comparison_data['Metric']):
                student_val = float(comparison_data['Student Model'][i])
                birdnet_val = float(comparison_data['BirdNET'][i])
                diff = birdnet_val - student_val
                diff_pct = (diff / student_val * 100) if student_val != 0 else 0
                differences.append(f"{diff:+.4f} ({diff_pct:+.1f}%)")
            
            comparison_data['Difference (BirdNET - Student)'] = differences
            
            # Save as CSV
            comparison_df = pd.DataFrame(comparison_data)
            csv_path = output_dir / 'metrics_comparison_table.csv'
            comparison_df.to_csv(csv_path, index=False)
            logger.info(f"Metrics comparison table saved to: {csv_path}")
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error saving metrics comparison table: {e}")
            return None

    def _save_per_class_metrics_table(self, results: Dict, output_dir: Path, naming_config: Dict = None):
        """Save detailed per-class metrics comparison table."""
        try:
            student_per_class = results['detailed_metrics']['student']['per_class']
            birdnet_per_class = results['detailed_metrics']['birdnet']['per_class']
            
            # Get classes (exclude summary entries)
            classes = [k for k in student_per_class.keys() 
                      if k not in ['accuracy', 'macro avg', 'weighted avg']]
            
            per_class_data = []
            
            for class_name in classes:
                if class_name in student_per_class and class_name in birdnet_per_class:
                    student_class = student_per_class[class_name]
                    birdnet_class = birdnet_per_class[class_name]
                    
                    per_class_data.append({
                        'Class': class_name,
                        'Support': int(student_class.get('support', 0)),
                        'Student_Precision': f"{student_class.get('precision', 0):.4f}",
                        'Student_Recall': f"{student_class.get('recall', 0):.4f}",
                        'Student_F1': f"{student_class.get('f1-score', 0):.4f}",
                        'BirdNET_Precision': f"{birdnet_class.get('precision', 0):.4f}",
                        'BirdNET_Recall': f"{birdnet_class.get('recall', 0):.4f}",
                        'BirdNET_F1': f"{birdnet_class.get('f1-score', 0):.4f}",
                        'Precision_Diff': f"{birdnet_class.get('precision', 0) - student_class.get('precision', 0):+.4f}",
                        'Recall_Diff': f"{birdnet_class.get('recall', 0) - student_class.get('recall', 0):+.4f}",
                        'F1_Diff': f"{birdnet_class.get('f1-score', 0) - student_class.get('f1-score', 0):+.4f}"
                    })
            
            # Save as CSV
            per_class_df = pd.DataFrame(per_class_data)
            csv_path = output_dir / 'per_class_metrics_table.csv'
            per_class_df.to_csv(csv_path, index=False)
            logger.info(f"Per-class metrics table saved to: {csv_path}")
            
            return per_class_df
            
        except Exception as e:
            logger.error(f"Error saving per-class metrics table: {e}")
            return None

    def _plot_comprehensive_metrics_comparison(self, results: Dict, output_dir: Path, plots_config: Dict = None):
        """Create comprehensive metrics comparison visualization."""
        try:
            student_metrics = results['detailed_metrics']['student']
            birdnet_metrics = results['detailed_metrics']['birdnet']
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Comprehensive Model Metrics Comparison', fontsize=16, fontweight='bold')
            
            # 1. Overall metrics comparison (top-left)
            metrics_names = ['Accuracy', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1-Score\n(Macro)']
            student_values = [
                student_metrics['accuracy'],
                student_metrics['precision_macro'],
                student_metrics['recall_macro'],
                student_metrics['f1_macro']
            ]
            birdnet_values = [
                birdnet_metrics['accuracy'],
                birdnet_metrics['precision_macro'],
                birdnet_metrics['recall_macro'],
                birdnet_metrics['f1_macro']
            ]
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            axes[0,0].bar(x - width/2, student_values, width, label='Student Model', alpha=0.8, color='blue')
            axes[0,0].bar(x + width/2, birdnet_values, width, label='BirdNET', alpha=0.8, color='orange')
            
            axes[0,0].set_xlabel('Metrics')
            axes[0,0].set_ylabel('Score')
            axes[0,0].set_title('Overall Metrics Comparison')
            axes[0,0].set_xticks(x)
            axes[0,0].set_xticklabels(metrics_names)
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].set_ylim(0, 1.0)
            
            # Add value labels on bars
            for i, (s_val, b_val) in enumerate(zip(student_values, birdnet_values)):
                axes[0,0].text(i - width/2, s_val + 0.01, f'{s_val:.3f}', ha='center', va='bottom', fontsize=9)
                axes[0,0].text(i + width/2, b_val + 0.01, f'{b_val:.3f}', ha='center', va='bottom', fontsize=9)
            
            # 2. Per-class F1 scores (top-right)
            student_per_class = student_metrics['per_class']
            birdnet_per_class = birdnet_metrics['per_class']
            
            classes = [k for k in student_per_class.keys() 
                      if k not in ['accuracy', 'macro avg', 'weighted avg'] and student_per_class[k].get('support', 0) > 0]
            
            if classes:
                student_f1_scores = [student_per_class[c].get('f1-score', 0) for c in classes]
                birdnet_f1_scores = [birdnet_per_class[c].get('f1-score', 0) for c in classes]
                
                x_classes = np.arange(len(classes))
                
                axes[0,1].bar(x_classes - width/2, student_f1_scores, width, label='Student Model', alpha=0.8, color='blue')
                axes[0,1].bar(x_classes + width/2, birdnet_f1_scores, width, label='BirdNET', alpha=0.8, color='orange')
                
                axes[0,1].set_xlabel('Classes')
                axes[0,1].set_ylabel('F1-Score')
                axes[0,1].set_title('Per-Class F1-Score Comparison')
                axes[0,1].set_xticks(x_classes)
                axes[0,1].set_xticklabels(classes, rotation=45, ha='right')
                axes[0,1].legend()
                axes[0,1].grid(True, alpha=0.3)
                axes[0,1].set_ylim(0, 1.0)
            
            # 3. Precision vs Recall scatter plot (bottom-left)
            if classes:
                student_precision = [student_per_class[c].get('precision', 0) for c in classes]
                student_recall = [student_per_class[c].get('recall', 0) for c in classes]
                birdnet_precision = [birdnet_per_class[c].get('precision', 0) for c in classes]
                birdnet_recall = [birdnet_per_class[c].get('recall', 0) for c in classes]
                
                axes[1,0].scatter(student_recall, student_precision, alpha=0.7, s=100, color='blue', label='Student Model')
                axes[1,0].scatter(birdnet_recall, birdnet_precision, alpha=0.7, s=100, color='orange', label='BirdNET')
                
                # Add class labels
                for i, class_name in enumerate(classes):
                    axes[1,0].annotate(class_name, (student_recall[i], student_precision[i]), 
                                     xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
                    axes[1,0].annotate(class_name, (birdnet_recall[i], birdnet_precision[i]), 
                                     xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
                
                axes[1,0].set_xlabel('Recall')
                axes[1,0].set_ylabel('Precision')
                axes[1,0].set_title('Precision vs Recall by Class')
                axes[1,0].legend()
                axes[1,0].grid(True, alpha=0.3)
                axes[1,0].set_xlim(0, 1.0)
                axes[1,0].set_ylim(0, 1.0)
            
            # 4. Model agreement summary (bottom-right)
            agreement = results['agreement_analysis']
            labels = ['Both\nCorrect', 'Student\nOnly', 'BirdNET\nOnly', 'Both\nIncorrect']
            values = [
                agreement['counts']['both_correct'],
                agreement['counts']['student_only_correct'],
                agreement['counts']['birdnet_only_correct'],
                agreement['counts']['both_incorrect']
            ]
            colors = ['green', 'blue', 'orange', 'red']
            
            bars = axes[1,1].bar(labels, values, color=colors, alpha=0.7)
            axes[1,1].set_title('Model Agreement Analysis')
            axes[1,1].set_ylabel('Number of Samples')
            axes[1,1].grid(True, alpha=0.3)
            
            # Add percentage labels on bars
            total_samples = sum(values)
            for bar, count in zip(bars, values):
                height = bar.get_height()
                percentage = count / total_samples * 100
                axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                              f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            filename = plots_config.get('comprehensive_metrics_comparison', 'comprehensive_metrics_comparison.png') if plots_config else 'comprehensive_metrics_comparison.png'
            plot_path = output_dir / filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Comprehensive metrics comparison saved to: {plot_path}")
            
        except Exception as e:
            logger.error(f"Error creating comprehensive metrics comparison: {e}")


@hydra.main(version_base=None, config_path="config", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    """Main function for model comparison."""
    
    # Get original working directory (before Hydra changes it)
    from hydra.core.hydra_config import HydraConfig
    original_cwd = HydraConfig.get().runtime.cwd
    
    logger.info("Starting model comparison")
    
    try:
        # Load predictions
        predictions_dir = os.path.join(original_cwd, cfg.benchmark.paths.predictions_dir)
        
        student_path = os.path.join(predictions_dir, "student_predictions.csv")
        birdnet_path = os.path.join(predictions_dir, "birdnet_predictions.csv")
        
        if not os.path.exists(student_path):
            logger.error(f"Student predictions not found: {student_path}")
            return
        
        if not os.path.exists(birdnet_path):
            logger.error(f"BirdNET predictions not found: {birdnet_path}")
            return
        
        student_df = pd.read_csv(student_path)
        birdnet_df = pd.read_csv(birdnet_path)
        
        logger.info(f"Loaded student predictions: {len(student_df)} rows")
        logger.info(f"Loaded BirdNET predictions: {len(birdnet_df)} rows")
        
        # Merge predictions
        merged_df = pd.merge(
            student_df[['audio_path', 'ground_truth', 'student_prediction', 'student_confidence']],
            birdnet_df[['audio_path', 'birdnet_prediction', 'birdnet_confidence']],
            on='audio_path',
            how='inner'
        )
        
        if len(merged_df) == 0:
            logger.error("No matching predictions found!")
            return
        
        logger.info(f"Merged predictions: {len(merged_df)} rows")
        
        # Initialize comparator - pass the full config for access to benchmark.mode
        comparator = ModelComparator(cfg)
        
        # Generate comparison
        results = comparator.compare_predictions(
            ground_truth=merged_df['ground_truth'].tolist(),
            student_predictions=merged_df['student_prediction'].tolist(),
            birdnet_predictions=merged_df['birdnet_prediction'].tolist(),
            student_confidences=merged_df['student_confidence'].tolist(),
            birdnet_confidences=merged_df['birdnet_confidence'].tolist(),
            audio_paths=merged_df['audio_path'].tolist()
        )
        
        # Save results
        output_dir = os.path.join(original_cwd, cfg.benchmark.paths.comparison_dir)
        comparator.save_results(results, output_dir)
        
        # Log summary
        metrics = results['metrics']
        logger.info("Comparison completed!")
        logger.info(f"Student Model - Accuracy: {metrics['student_accuracy']:.4f}, F1: {metrics['student_f1']:.4f}")
        logger.info(f"BirdNET - Accuracy: {metrics['birdnet_accuracy']:.4f}, F1: {metrics['birdnet_f1']:.4f}")
        
    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        raise


if __name__ == "__main__":
    main() 