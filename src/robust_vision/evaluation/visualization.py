"""Visualization utilities for robustness evaluation."""

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path


# Set publication-quality style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_robustness_curves(
    results: Dict[str, List[Dict[str, float]]],
    metric: str = 'accuracy',
    output_path: Optional[str] = None,
    figsize: tuple = (12, 5)
):
    """
    Plot robustness curves for different noise types.
    
    Args:
        results: Results dictionary from RobustnessEvaluator
        metric: Metric to plot ('accuracy', 'mean_confidence', 'mean_margin')
        output_path: Optional path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: All noise types together
    ax1 = axes[0]
    for noise_type, metrics_list in results.items():
        if noise_type == 'clean':
            continue
        severities = [m['severity'] for m in metrics_list]
        values = [m[metric] for m in metrics_list]
        ax1.plot(severities, values, marker='o', label=noise_type.replace('_', ' ').title())
    
    ax1.set_xlabel('Noise Severity')
    ax1.set_ylabel(metric.replace('_', ' ').title())
    ax1.set_title(f'Robustness Curves - {metric.replace("_", " ").title()}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Individual noise types (subplots)
    ax2 = axes[1]
    
    # Calculate degradation (relative to clean)
    if 'clean' in results and len(results['clean']) > 0:
        clean_value = results['clean'][0][metric]
        
        for noise_type, metrics_list in results.items():
            if noise_type == 'clean':
                continue
            severities = [m['severity'] for m in metrics_list]
            degradation = [(clean_value - m[metric]) / clean_value * 100 for m in metrics_list]
            ax2.plot(severities, degradation, marker='s', label=noise_type.replace('_', ' ').title())
        
        ax2.set_xlabel('Noise Severity')
        ax2.set_ylabel(f'{metric.replace("_", " ").title()} Degradation (%)')
        ax2.set_title(f'Relative Degradation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Robustness curves saved to {output_path}")
    
    return fig


def plot_margin_distribution(
    logits: np.ndarray,
    labels: np.ndarray,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    Plot distribution of prediction margins.
    
    Args:
        logits: Model logits [batch, num_classes]
        labels: Ground truth labels [batch]
        output_path: Optional path to save figure
        figsize: Figure size
    """
    # Compute margins (difference between correct class and max incorrect class)
    margins = []
    for i in range(len(logits)):
        correct_logit = logits[i, labels[i]]
        incorrect_logits = np.delete(logits[i], labels[i])
        max_incorrect = np.max(incorrect_logits)
        margin = correct_logit - max_incorrect
        margins.append(margin)
    
    margins = np.array(margins)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Histogram
    ax1 = axes[0]
    ax1.hist(margins, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(margins), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(margins):.2f}')
    ax1.axvline(np.median(margins), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(margins):.2f}')
    ax1.set_xlabel('Margin (Correct - Max Incorrect)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Prediction Margins')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative distribution
    ax2 = axes[1]
    sorted_margins = np.sort(margins)
    cumulative = np.arange(1, len(sorted_margins) + 1) / len(sorted_margins)
    ax2.plot(sorted_margins, cumulative, linewidth=2)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero Margin')
    ax2.set_xlabel('Margin (Correct - Max Incorrect)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution of Margins')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Margin distribution saved to {output_path}")
    
    return fig


def plot_training_history(
    metrics_file: str,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 5)
):
    """
    Plot training history from metrics file.
    
    Args:
        metrics_file: Path to JSONL metrics file
        output_path: Optional path to save figure
        figsize: Figure size
    """
    # Read metrics
    import json
    metrics = []
    with open(metrics_file, 'r') as f:
        for line in f:
            metrics.append(json.loads(line))
    
    df = pd.DataFrame(metrics)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Loss
    ax1 = axes[0]
    if 'loss' in df.columns:
        ax1.plot(df['step'], df['loss'], marker='o', markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    ax2 = axes[1]
    if 'accuracy' in df.columns:
        ax2.plot(df['step'], df['accuracy'], marker='o', markersize=3, color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Training history saved to {output_path}")
    
    return fig


def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 8)
):
    """
    Plot confusion matrix.
    
    Args:
        predictions: Predicted labels [batch]
        labels: Ground truth labels [batch]
        class_names: Optional list of class names
        output_path: Optional path to save figure
        figsize: Figure size
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(labels, predictions)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    
    return fig


def create_robustness_report(
    results: Dict[str, List[Dict[str, float]]],
    output_dir: str = "./results"
):
    """
    Create a comprehensive robustness report with multiple visualizations.
    
    Args:
        results: Results dictionary from RobustnessEvaluator
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating robustness report...")
    
    # Plot accuracy curves
    plot_robustness_curves(
        results,
        metric='accuracy',
        output_path=str(output_path / "robustness_accuracy.png")
    )
    plt.close()
    
    # Plot confidence curves
    plot_robustness_curves(
        results,
        metric='mean_confidence',
        output_path=str(output_path / "robustness_confidence.png")
    )
    plt.close()
    
    # Plot margin curves
    plot_robustness_curves(
        results,
        metric='mean_margin',
        output_path=str(output_path / "robustness_margin.png")
    )
    plt.close()
    
    print(f"Robustness report generated in {output_dir}")
