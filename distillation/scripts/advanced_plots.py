#!/usr/bin/env python3
"""
Advanced plotting utilities for knowledge distillation training analysis.
Generates detailed visualizations of filter parameters evolution and training dynamics.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from typing import Dict, List, Optional, Tuple

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DistillationAnalyzer:
    """Advanced analysis and plotting for distillation training results."""
    
    def __init__(self, log_dir: str):
        """Initialize analyzer with training log directory."""
        self.log_dir = Path(log_dir)
        self.data = {}
        self.load_training_data()
    
    def load_training_data(self):
        """Load all training data from log files."""
        print(f"Loading training data from {self.log_dir}")
        
        # Load filter parameters evolution
        filter_csv = self.log_dir / "filter_parameters_evolution.csv"
        if filter_csv.exists():
            self.data['filter_params'] = pd.read_csv(filter_csv)
            print(f"✅ Loaded filter parameters: {len(self.data['filter_params'])} epochs")
        
        # Load results JSON
        results_json = self.log_dir / "results.json"
        if results_json.exists():
            with open(results_json, 'r') as f:
                self.data['results'] = json.load(f)
            print(f"✅ Loaded results: Test accuracy = {self.data['results']['test_acc']:.4f}")
            
        # Parse training log
        self.parse_training_log()
    
    def parse_training_log(self):
        """Parse training log to extract loss and accuracy evolution."""
        log_file = self.log_dir / "distillation_train.log"
        if not log_file.exists():
            print("❌ Training log not found")
            return
            
        epochs = []
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        with open(log_file, 'r') as f:
            for line in f:
                if "Epoch" in line and "[Train]" in line and "Avg Loss:" in line:
                    # Format: "2025-06-17 09:42:25,024 - [INFO] - Epoch 1 [Train] Avg Loss: 1.1127, Avg Acc: 42.7605"
                    parts = line.split()
                    epoch_idx = parts.index("Epoch") + 1
                    epoch = int(parts[epoch_idx])
                    loss_idx = parts.index("Loss:") + 1
                    acc_idx = parts.index("Acc:") + 1
                    
                    train_loss = float(parts[loss_idx].rstrip(','))
                    train_acc = float(parts[acc_idx])
                    
                    epochs.append(epoch)
                    train_losses.append(train_loss)
                    train_accs.append(train_acc)
                
                elif "Epoch" in line and "[Val]" in line and "Avg Loss:" in line:
                    # Format: "2025-06-17 09:43:31,970 - [INFO] - Epoch 1 [Val] Avg Loss: 0.8522, Avg Acc: 68.3398"
                    parts = line.split()
                    loss_idx = parts.index("Loss:") + 1
                    acc_idx = parts.index("Acc:") + 1
                    
                    val_loss = float(parts[loss_idx].rstrip(','))
                    val_acc = float(parts[acc_idx])
                    
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)
        
        self.data['training'] = {
            'epochs': epochs,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
        
        print(f"✅ Parsed training log: {len(epochs)} epochs")
    
    def plot_filter_evolution(self, save_path: Optional[str] = None):
        """Plot comprehensive filter parameter evolution analysis."""
        if 'filter_params' not in self.data:
            print("❌ No filter parameters data available")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🎛️ Differentiable Filter Evolution Analysis', fontsize=16)
        
        df = self.data['filter_params']
        epochs = df['epoch'].values
        breakpoints = df['breakpoint_hz'].values
        transition_widths = df['transition_width'].values
        
        # Plot 1: Dual-axis parameter evolution
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(epochs, breakpoints, 'g.-', linewidth=2, markersize=8, label='Breakpoint')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Breakpoint (Hz)', color='g')
        ax1.tick_params(axis='y', labelcolor='g')
        ax1.grid(True, alpha=0.3)
        
        line2 = ax1_twin.plot(epochs, transition_widths, 'r.-', linewidth=2, markersize=8, label='Transition Width')
        ax1_twin.set_ylabel('Transition Width', color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        ax1.set_title('Filter Parameters Evolution')
        
        # Plot 2: Convergence speed analysis
        ax2 = axes[0, 1]
        if len(epochs) > 1:
            breakpoint_changes = np.abs(np.diff(breakpoints))
            transition_changes = np.abs(np.diff(transition_widths))
            
            ax2.semilogy(epochs[1:], breakpoint_changes, 'g.-', linewidth=2, label='|Δ Breakpoint|')
            ax2.semilogy(epochs[1:], transition_changes, 'r.-', linewidth=2, label='|Δ Transition Width|')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Absolute Change (log scale)')
            ax2.set_title('Parameter Convergence Speed')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Need >1 epoch\nfor convergence analysis', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # Plot 3: Filter-performance correlation
        ax3 = axes[1, 0]
        if 'training' in self.data and len(self.data['training']['val_accs']) == len(breakpoints):
            val_accs = np.array(self.data['training']['val_accs'])
            scatter = ax3.scatter(breakpoints, val_accs, c=epochs, cmap='viridis', 
                                s=100, alpha=0.7, edgecolors='black')
            ax3.set_xlabel('Breakpoint (Hz)')
            ax3.set_ylabel('Validation Accuracy (%)')
            ax3.set_title('Filter vs Performance Correlation')
            ax3.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('Epoch')
            
            if len(breakpoints) > 2:
                z = np.polyfit(breakpoints, val_accs, 1)
                p = np.poly1d(z)
                ax3.plot(breakpoints, p(breakpoints), 'r--', alpha=0.8, 
                        label=f'Trend (slope={z[0]:.2e})')
                ax3.legend()
        
        # Plot 4: Parameter statistics summary
        ax4 = axes[1, 1]
        stats_data = {
            'Breakpoint': {
                'Initial': breakpoints[0],
                'Final': breakpoints[-1],
                'Change': breakpoints[-1] - breakpoints[0],
                'Range': np.ptp(breakpoints)
            },
            'Transition Width': {
                'Initial': transition_widths[0],
                'Final': transition_widths[-1],
                'Change': transition_widths[-1] - transition_widths[0],
                'Range': np.ptp(transition_widths)
            }
        }
        
        # Create grouped bar chart
        x = np.arange(4)
        width = 0.35
        
        bp_values = list(stats_data['Breakpoint'].values())
        tw_values = list(stats_data['Transition Width'].values())
        
        # Normalize transition width values for comparison
        tw_normalized = np.array(tw_values) * (np.mean(bp_values) / np.mean(tw_values))
        
        bars1 = ax4.bar(x - width/2, bp_values, width, label='Breakpoint (Hz)', alpha=0.7)
        bars2 = ax4.bar(x + width/2, tw_normalized, width, label='Trans. Width (normalized)', alpha=0.7)
        
        ax4.set_xlabel('Statistic')
        ax4.set_ylabel('Value')
        ax4.set_title('Filter Parameters Summary')
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Initial', 'Final', 'Change', 'Range'])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved filter evolution plot to {save_path}")
        
        return fig
    
    def plot_training_dynamics(self, save_path: Optional[str] = None):
        """Plot comprehensive training dynamics analysis."""
        if 'training' not in self.data:
            print("❌ No training data available")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('📊 Training Dynamics Deep Analysis', fontsize=16)
        
        training = self.data['training']
        epochs = np.array(training['epochs'])
        train_losses = np.array(training['train_losses'])
        val_losses = np.array(training['val_losses'])
        train_accs = np.array(training['train_accs'])
        val_accs = np.array(training['val_accs'])
        
        # Plot 1: Loss evolution with trend analysis
        ax1 = axes[0, 0]
        ax1.plot(epochs, train_losses, 'b.-', linewidth=2, label='Train Loss', markersize=6)
        ax1.plot(epochs, val_losses, 'r.-', linewidth=2, label='Val Loss', markersize=6)
        
        if len(epochs) > 2:
            train_trend = np.polyfit(epochs, train_losses, 1)
            val_trend = np.polyfit(epochs, val_losses, 1)
            ax1.plot(epochs, np.poly1d(train_trend)(epochs), 'b--', alpha=0.5, 
                    label=f'Train trend ({train_trend[0]:.3f})')
            ax1.plot(epochs, np.poly1d(val_trend)(epochs), 'r--', alpha=0.5,
                    label=f'Val trend ({val_trend[0]:.3f})')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Evolution with Trends')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy evolution with confidence intervals
        ax2 = axes[0, 1]
        ax2.plot(epochs, train_accs, 'g.-', linewidth=2, label='Train Accuracy', markersize=6)
        ax2.plot(epochs, val_accs, 'orange', marker='.', linestyle='-', linewidth=2, 
                label='Val Accuracy', markersize=6)
        
        # Add smoothed trend lines
        if len(epochs) > 3:
            try:
                from scipy.ndimage import gaussian_filter1d
                smooth_train = gaussian_filter1d(train_accs, sigma=0.5)
                smooth_val = gaussian_filter1d(val_accs, sigma=0.5)
                ax2.plot(epochs, smooth_train, 'g--', alpha=0.5, label='Train (smoothed)')
                ax2.plot(epochs, smooth_val, '--', color='orange', alpha=0.5, label='Val (smoothed)')
            except ImportError:
                # Fallback to simple moving average if scipy not available
                window = min(3, len(epochs))
                smooth_train = np.convolve(train_accs, np.ones(window)/window, mode='same')
                smooth_val = np.convolve(val_accs, np.ones(window)/window, mode='same')
                ax2.plot(epochs, smooth_train, 'g--', alpha=0.5, label='Train (smoothed)')
                ax2.plot(epochs, smooth_val, '--', color='orange', alpha=0.5, label='Val (smoothed)')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Generalization gap analysis
        ax3 = axes[0, 2]
        generalization_gap = train_accs - val_accs
        ax3.plot(epochs, generalization_gap, 'purple', marker='o', linewidth=2, 
                label='Generalization Gap', markersize=6)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.fill_between(epochs, generalization_gap, alpha=0.3, color='purple')
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Train Acc - Val Acc (%)')
        ax3.set_title('Overfitting Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add interpretation
        avg_gap = np.mean(generalization_gap)
        if avg_gap > 5:
            ax3.text(0.05, 0.95, 'Potential\nOverfitting', transform=ax3.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
        elif avg_gap < -2:
            ax3.text(0.05, 0.95, 'Underfitting', transform=ax3.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.3))
        else:
            ax3.text(0.05, 0.95, 'Good\nGeneralization', transform=ax3.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.3))
        
        # Plot 4: Learning rate vs performance (if available)
        ax4 = axes[1, 0]
        if len(epochs) > 1:
            loss_improvements = -np.diff(val_losses)
            acc_improvements = np.diff(val_accs)
            
            ax4.bar(epochs[1:] - 0.2, loss_improvements, width=0.4, alpha=0.7, 
                   label='Loss Improvement', color='blue')
            ax4_twin = ax4.twinx()
            ax4_twin.bar(epochs[1:] + 0.2, acc_improvements, width=0.4, alpha=0.7,
                        label='Acc Improvement', color='green')
            
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss Improvement', color='blue')
            ax4_twin.set_ylabel('Accuracy Improvement (%)', color='green')
            ax4.set_title('Learning Efficiency per Epoch')
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Performance correlation heatmap
        ax5 = axes[1, 1]
        if 'filter_params' in self.data and len(self.data['filter_params']) == len(epochs):
            df_corr = pd.DataFrame({
                'Train_Loss': train_losses,
                'Val_Loss': val_losses,
                'Train_Acc': train_accs,
                'Val_Acc': val_accs,
                'Breakpoint': self.data['filter_params']['breakpoint_hz'].values,
                'Trans_Width': self.data['filter_params']['transition_width'].values
            })
            
            corr_matrix = df_corr.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax5, cbar_kws={'shrink': 0.8})
            ax5.set_title('Performance Correlation Matrix')
        
        # Plot 6: Training statistics table
        ax6 = axes[1, 2]
        stats = {
            'Best Val Acc': f"{max(val_accs):.2f}%",
            'Final Val Acc': f"{val_accs[-1]:.2f}%",
            'Best Train Acc': f"{max(train_accs):.2f}%",
            'Final Train Acc': f"{train_accs[-1]:.2f}%",
            'Min Val Loss': f"{min(val_losses):.4f}",
            'Final Val Loss': f"{val_losses[-1]:.4f}",
            'Avg Gen. Gap': f"{np.mean(generalization_gap):.2f}%",
            'Total Epochs': f"{len(epochs)}"
        }
        
        ax6.axis('off')
        table_text = [[key, value] for key, value in stats.items()]
        
        table = ax6.table(cellText=table_text, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        ax6.set_title('Training Summary Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved training dynamics plot to {save_path}")
        
        return fig
    
    def generate_all_plots(self, output_dir: Optional[str] = None):
        """Generate all available plots and save them."""
        if output_dir is None:
            output_dir = self.log_dir / "advanced_plots"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        print(f"🎨 Generating advanced plots in {output_dir}")
        
        plots = [
            ("filter_evolution_analysis.png", self.plot_filter_evolution),
            ("training_dynamics_analysis.png", self.plot_training_dynamics)
        ]
        
        generated_plots = []
        for filename, plot_func in plots:
            try:
                save_path = output_dir / filename
                fig = plot_func(save_path)
                if fig is not None:
                    plt.close(fig)
                    generated_plots.append(save_path)
            except Exception as e:
                print(f"❌ Error generating {filename}: {e}")
        
        print(f"✅ Generated {len(generated_plots)} advanced plots")
        return generated_plots


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Generate advanced plots for distillation training analysis')
    parser.add_argument('log_dir', help='Path to training log directory')
    parser.add_argument('--output-dir', help='Output directory for plots (default: log_dir/advanced_plots)')
    parser.add_argument('--plot-type', choices=['filter', 'dynamics', 'all'], 
                       default='all', help='Type of plot to generate')
    
    args = parser.parse_args()
    
    analyzer = DistillationAnalyzer(args.log_dir)
    
    if args.plot_type == 'all':
        analyzer.generate_all_plots(args.output_dir)
    elif args.plot_type == 'filter':
        save_path = Path(args.output_dir or analyzer.log_dir) / "filter_evolution_analysis.png"
        analyzer.plot_filter_evolution(save_path)
    elif args.plot_type == 'dynamics':
        save_path = Path(args.output_dir or analyzer.log_dir) / "training_dynamics_analysis.png"
        analyzer.plot_training_dynamics(save_path)
    
    print("🎨 Advanced plotting completed!")


if __name__ == "__main__":
    main() 