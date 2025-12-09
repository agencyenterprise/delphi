#!/usr/bin/env python3
"""
Create a 2D histogram heatmap comparing baseline F1 scores with best magic_planet F1 scores.

X-axis: Baseline auto-interp label F1 score
Y-axis: Best F1 score across all magic_planet labels/scales
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import sys
from collections import defaultdict


def load_results(filepath):
    """Load the detailed results JSON file."""
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries")
    return data


def extract_baseline_f1(baseline_data):
    """
    Extract F1 scores from baseline data.
    
    Returns:
        Dict mapping latent_index -> f1_score
    """
    f1_scores = {}
    for entry in baseline_data:
        latent_idx = entry['latent_index']
        f1 = entry['metrics']['f1']
        f1_scores[latent_idx] = f1
    return f1_scores


def extract_best_f1(magic_planet_data):
    """
    Extract the best F1 score for each latent from magic_planet data.
    
    Returns:
        Dict mapping latent_index -> best_f1_score
    """
    # Group by latent
    grouped = defaultdict(list)
    for entry in magic_planet_data:
        latent_idx = entry['latent_index']
        grouped[latent_idx].append(entry)
    
    # Find best F1 for each latent
    best_f1_scores = {}
    for latent_idx, entries in grouped.items():
        best_f1 = max(entry['metrics']['f1'] for entry in entries)
        best_f1_scores[latent_idx] = best_f1
    
    return best_f1_scores


def create_heatmap(baseline_f1, best_f1, output_path, n_bins=39):
    """
    Create a 2D histogram heatmap comparing baseline vs best F1 scores.
    
    Args:
        baseline_f1: Dict mapping latent_index -> baseline_f1
        best_f1: Dict mapping latent_index -> best_f1
        output_path: Path to save the plot
        n_bins: Number of bins for the histogram (default: 39)
    """
    # Find common latent indices
    common_latents = sorted(set(baseline_f1.keys()) & set(best_f1.keys()))
    
    if not common_latents:
        print("ERROR: No common latent indices found!")
        return
    
    print(f"\nFound {len(common_latents)} common latent indices")
    
    # Extract matched pairs
    x_vals = np.array([baseline_f1[idx] for idx in common_latents])
    y_vals = np.array([best_f1[idx] for idx in common_latents])
    
    # Calculate statistics
    improvements = y_vals - x_vals
    mean_improvement = np.mean(improvements)
    median_improvement = np.median(improvements)
    above_diagonal = np.sum(improvements > 0)
    below_diagonal = np.sum(improvements < 0)
    on_diagonal = np.sum(improvements == 0)
    
    # Calculate baseline and best means
    baseline_mean = np.mean(x_vals)
    best_mean = np.mean(y_vals)
    
    print(f"\nStatistics:")
    print(f"  Baseline mean F1: {baseline_mean:.4f}")
    print(f"  Best mean F1: {best_mean:.4f}")
    print(f"  Mean improvement: {mean_improvement:.4f}")
    print(f"  Median improvement: {median_improvement:.4f}")
    print(f"  Points above diagonal (improved): {above_diagonal} ({100*above_diagonal/len(common_latents):.1f}%)")
    print(f"  Points below diagonal (worse): {below_diagonal} ({100*below_diagonal/len(common_latents):.1f}%)")
    print(f"  Points on diagonal (same): {on_diagonal}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 11))
    
    # Create 2D histogram
    bins = np.linspace(0, 1, n_bins + 1)
    h, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=bins)
    
    # Plot heatmap with logarithmic color scale
    # Add small epsilon to avoid log(0)
    h_plot = h.T.copy()
    h_plot[h_plot == 0] = 0.5  # Set empty bins to 0.5 so they show up as very faint
    
    im = ax.imshow(h_plot, origin='lower', extent=[0, 1, 0, 1], 
                   cmap='viridis', aspect='auto', interpolation='nearest',
                   norm=LogNorm(vmin=0.5, vmax=h_plot.max()))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Number of latents')
    
    # Add diagonal line (y=x)
    ax.plot([0, 1], [0, 1], 'b--', alpha=0.7, linewidth=2, label='y=x (no improvement)')
    
    # Add grid aligned with bins
    for edge in bins:
        ax.axhline(y=edge, color='white', alpha=0.2, linewidth=0.5)
        ax.axvline(x=edge, color='white', alpha=0.2, linewidth=0.5)
    
    # Labels and title
    ax.set_xlabel('Baseline F1 Score (auto-interp label)', fontsize=13)
    ax.set_ylabel('Best F1 Score (magic_planet_51)', fontsize=13)
    ax.set_title('Baseline vs Best Magic Planet F1 Scores (2D Histogram)\n' + 
                 f'Baseline mean: {baseline_mean:.4f} | Best mean: {best_mean:.4f} | ' +
                 f'{above_diagonal}/{len(common_latents)} improved ({100*above_diagonal/len(common_latents):.1f}%)',
                 fontsize=12)
    
    # Set limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add legend
    ax.legend(loc='upper left')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")
    
    return {
        'n_latents': len(common_latents),
        'baseline_mean_f1': float(baseline_mean),
        'best_mean_f1': float(best_mean),
        'mean_improvement': float(mean_improvement),
        'median_improvement': float(median_improvement),
        'n_improved': int(above_diagonal),
        'n_worse': int(below_diagonal),
        'n_same': int(on_diagonal),
        'pct_improved': float(100 * above_diagonal / len(common_latents))
    }


def main():
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python plot_baseline_vs_best_heatmap.py <baseline_results.json> <magic_planet_results.json> [output_path] [n_bins]")
        sys.exit(1)
    
    baseline_file = Path(sys.argv[1])
    magic_planet_file = Path(sys.argv[2])
    
    # Determine output path
    if len(sys.argv) >= 4:
        output_path = Path(sys.argv[3])
    else:
        output_path = magic_planet_file.parent / 'baseline_vs_best_f1_heatmap.png'
    
    # Determine number of bins
    n_bins = int(sys.argv[4]) if len(sys.argv) >= 5 else 39
    
    print(f"Using {n_bins} bins for the histogram")
    
    # Load data
    baseline_data = load_results(baseline_file)
    magic_planet_data = load_results(magic_planet_file)
    
    # Extract F1 scores
    print("\nExtracting baseline F1 scores...")
    baseline_f1 = extract_baseline_f1(baseline_data)
    print(f"Found {len(baseline_f1)} baseline F1 scores")
    
    print("\nExtracting best F1 scores from magic_planet...")
    best_f1 = extract_best_f1(magic_planet_data)
    print(f"Found {len(best_f1)} best F1 scores")
    
    # Create heatmap
    print("\nCreating heatmap...")
    stats = create_heatmap(baseline_f1, best_f1, output_path, n_bins)
    
    # Save statistics
    stats_path = output_path.parent / 'baseline_vs_best_heatmap_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {stats_path}")


if __name__ == '__main__':
    main()

