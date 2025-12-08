#!/usr/bin/env python3
"""
Analyze scoring results and generate comparison visualizations.

Usage:
    python analyze_scoring_results.py results/baseline_detection
    python analyze_scoring_results.py results/experimental_magic_planet_detection --compare results/baseline_detection
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_results(results_dir: Path) -> tuple[list, dict]:
    """Load detailed results and summary statistics."""
    detailed = json.load(open(results_dir / "detailed_results.json"))
    summary = json.load(open(results_dir / "summary_statistics.json"))
    return detailed, summary


def print_summary(summary: dict, name: str = "Results"):
    """Print formatted summary statistics."""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    print(f"Total labels scored: {summary['total_labels']}")
    print(f"Unique latents: {summary['unique_latents']}")
    
    overall = summary['overall']
    print(f"\nOverall Performance:")
    print(f"  F1 Score:   {overall['mean_f1']:.4f} ± {overall['std_f1']:.4f}")
    print(f"  Accuracy:   {overall['mean_accuracy']:.4f} ± {overall['std_accuracy']:.4f}")
    print(f"  Precision:  {overall['mean_precision']:.4f} ± {overall['std_precision']:.4f}")
    print(f"  Recall:     {overall['mean_recall']:.4f} ± {overall['std_recall']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TP: {overall['total_tp']:>6}  |  FP: {overall['total_fp']:>6}")
    print(f"  FN: {overall['total_fn']:>6}  |  TN: {overall['total_tn']:>6}")
    
    if "by_scale" in summary:
        print(f"\nPer-Scale Performance:")
        print(f"  {'Scale':<8} {'N':<6} {'F1':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12}")
        print(f"  {'-'*70}")
        for scale, stats in sorted(summary["by_scale"].items(), key=lambda x: float(x[0])):
            print(f"  {scale:<8} {stats['n_labels']:<6} "
                  f"{stats['mean_f1']:.4f}±{stats['std_f1']:.3f}  "
                  f"{stats['mean_accuracy']:.4f}±{stats['std_accuracy']:.3f}  "
                  f"{stats['mean_precision']:.4f}  "
                  f"{stats['mean_recall']:.4f}")


def plot_f1_distribution(detailed: list, output_dir: Path, title: str = "F1 Score Distribution"):
    """Plot histogram of F1 scores across all labels."""
    f1_scores = [r['metrics']['f1'] for r in detailed]
    
    plt.figure(figsize=(10, 6))
    plt.hist(f1_scores, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(f1_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(f1_scores):.3f}')
    plt.axvline(np.median(f1_scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(f1_scores):.3f}')
    plt.xlabel('F1 Score')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "f1_distribution.png", dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_dir / 'f1_distribution.png'}")


def plot_scale_comparison(detailed: list, output_dir: Path):
    """Plot F1 scores across different scales (for experimental conditions)."""
    # Check if we have scale data
    if detailed[0].get('scale') is None:
        return
    
    # Organize by scale
    df = pd.DataFrame([
        {
            'scale': r['scale'],
            'f1': r['metrics']['f1'],
            'accuracy': r['metrics']['accuracy'],
            'precision': r['metrics']['precision'],
            'recall': r['metrics']['recall'],
        }
        for r in detailed
    ])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['f1', 'accuracy', 'precision', 'recall']
    titles = ['F1 Score', 'Accuracy', 'Precision', 'Recall']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        sns.boxplot(data=df, x='scale', y=metric, ax=ax)
        ax.set_xlabel('Scale')
        ax.set_ylabel(title)
        ax.set_title(f'{title} by Scale')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "scale_comparison.png", dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_dir / 'scale_comparison.png'}")


def plot_metric_correlation(detailed: list, output_dir: Path):
    """Plot correlation between different metrics."""
    df = pd.DataFrame([
        {
            'F1': r['metrics']['f1'],
            'Accuracy': r['metrics']['accuracy'],
            'Precision': r['metrics']['precision'],
            'Recall': r['metrics']['recall'],
            'Latent Frequency': r['latent_metadata']['per_token_frequency'],
        }
        for r in detailed
    ])
    
    # Correlation matrix
    corr = df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Metric Correlations')
    plt.tight_layout()
    plt.savefig(output_dir / "metric_correlations.png", dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_dir / 'metric_correlations.png'}")


def compare_conditions(
    detailed1: list,
    summary1: dict,
    name1: str,
    detailed2: Optional[list] = None,
    summary2: Optional[dict] = None,
    name2: Optional[str] = None,
    output_dir: Optional[Path] = None,
):
    """Compare two different conditions (e.g., baseline vs experimental)."""
    if detailed2 is None or summary2 is None:
        return
    
    print(f"\n{'='*80}")
    print(f"Comparison: {name1} vs {name2}")
    print(f"{'='*80}")
    
    # Overall comparison
    print(f"\n{'Metric':<15} {name1:<20} {name2:<20} {'Difference':<15}")
    print(f"{'-'*70}")
    
    metrics = ['mean_f1', 'mean_accuracy', 'mean_precision', 'mean_recall']
    for metric in metrics:
        val1 = summary1['overall'][metric]
        val2 = summary2['overall'][metric]
        diff = val2 - val1
        sign = '+' if diff >= 0 else ''
        print(f"{metric:<15} {val1:.4f}{' '*15} {val2:.4f}{' '*15} {sign}{diff:.4f}")
    
    # If both have scales, compare across scales
    if "by_scale" in summary1 and "by_scale" in summary2:
        print(f"\nPer-Scale F1 Comparison:")
        print(f"  {'Scale':<8} {name1:<12} {name2:<12} {'Difference':<12}")
        print(f"  {'-'*50}")
        
        scales1 = set(summary1["by_scale"].keys())
        scales2 = set(summary2["by_scale"].keys())
        common_scales = sorted(scales1 & scales2, key=float)
        
        for scale in common_scales:
            f1_1 = summary1["by_scale"][scale]["mean_f1"]
            f1_2 = summary2["by_scale"][scale]["mean_f1"]
            diff = f1_2 - f1_1
            sign = '+' if diff >= 0 else ''
            print(f"  {scale:<8} {f1_1:.4f}       {f1_2:.4f}       {sign}{diff:.4f}")
    
    # Plot side-by-side F1 distributions
    if output_dir:
        f1_scores1 = [r['metrics']['f1'] for r in detailed1]
        f1_scores2 = [r['metrics']['f1'] for r in detailed2]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.hist(f1_scores1, bins=50, edgecolor='black', alpha=0.7, color='blue')
        ax1.axvline(np.mean(f1_scores1), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(f1_scores1):.3f}')
        ax1.set_xlabel('F1 Score')
        ax1.set_ylabel('Count')
        ax1.set_title(name1)
        ax1.legend()
        
        ax2.hist(f1_scores2, bins=50, edgecolor='black', alpha=0.7, color='green')
        ax2.axvline(np.mean(f1_scores2), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(f1_scores2):.3f}')
        ax2.set_xlabel('F1 Score')
        ax2.set_ylabel('Count')
        ax2.set_title(name2)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "comparison_f1_distributions.png", dpi=150)
        plt.close()
        print(f"\n  ✓ Saved {output_dir / 'comparison_f1_distributions.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze scoring results")
    parser.add_argument("results_dir", type=Path, help="Results directory")
    parser.add_argument("--compare", type=Path, help="Compare with another results directory")
    parser.add_argument("--name", type=str, default="Primary", help="Name for primary results")
    parser.add_argument("--compare-name", type=str, default="Comparison", help="Name for comparison results")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    
    args = parser.parse_args()
    
    # Load primary results
    detailed, summary = load_results(args.results_dir)
    print_summary(summary, args.name)
    
    # Generate plots
    if not args.no_plots:
        print(f"\nGenerating plots...")
        plot_f1_distribution(detailed, args.results_dir, f"F1 Distribution - {args.name}")
        plot_scale_comparison(detailed, args.results_dir)
        plot_metric_correlation(detailed, args.results_dir)
    
    # Load and compare with second condition
    if args.compare:
        detailed2, summary2 = load_results(args.compare)
        print_summary(summary2, args.compare_name)
        
        output_dir = args.results_dir if not args.no_plots else None
        compare_conditions(
            detailed, summary, args.name,
            detailed2, summary2, args.compare_name,
            output_dir
        )


if __name__ == "__main__":
    main()

