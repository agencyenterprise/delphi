#!/usr/bin/env python3
"""
Create a scatter plot comparing baseline F1 scores with best magic_planet F1 scores.

X-axis: Baseline auto-interp label F1 score
Y-axis: Best F1 score across all magic_planet labels/scales
"""

import json
import numpy as np
import matplotlib.pyplot as plt
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


def create_scatter_plot(baseline_f1, best_f1, output_path):
    """
    Create a scatter plot comparing baseline vs best F1 scores.
    
    Args:
        baseline_f1: Dict mapping latent_index -> baseline_f1
        best_f1: Dict mapping latent_index -> best_f1
        output_path: Path to save the plot
    """
    # Find common latent indices
    common_latents = sorted(set(baseline_f1.keys()) & set(best_f1.keys()))
    
    if not common_latents:
        print("ERROR: No common latent indices found!")
        return
    
    print(f"\nFound {len(common_latents)} common latent indices")
    
    # Extract matched pairs
    x_vals = [baseline_f1[idx] for idx in common_latents]
    y_vals = [best_f1[idx] for idx in common_latents]
    
    # Calculate statistics
    improvements = [y - x for x, y in zip(x_vals, y_vals)]
    mean_improvement = np.mean(improvements)
    median_improvement = np.median(improvements)
    above_diagonal = sum(1 for imp in improvements if imp > 0)
    below_diagonal = sum(1 for imp in improvements if imp < 0)
    on_diagonal = sum(1 for imp in improvements if imp == 0)
    
    print(f"\nStatistics:")
    print(f"  Mean improvement: {mean_improvement:.4f}")
    print(f"  Median improvement: {median_improvement:.4f}")
    print(f"  Points above diagonal (improved): {above_diagonal} ({100*above_diagonal/len(common_latents):.1f}%)")
    print(f"  Points below diagonal (worse): {below_diagonal} ({100*below_diagonal/len(common_latents):.1f}%)")
    print(f"  Points on diagonal (same): {on_diagonal}")
    
    # Create the plot
    plt.figure(figsize=(10, 10))
    
    # Scatter plot
    plt.scatter(x_vals, y_vals, alpha=0.5, s=20, edgecolors='none')
    
    # Add diagonal line (y=x)
    min_val = min(min(x_vals), min(y_vals))
    max_val = max(max(x_vals), max(y_vals))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=1, label='y=x (no improvement)')
    
    # Labels and title
    plt.xlabel('Baseline F1 Score (auto-interp label)', fontsize=12)
    plt.ylabel('Best F1 Score (magic_planet_51)', fontsize=12)
    plt.title('Baseline vs Best Magic Planet F1 Scores\n' + 
              f'Mean Δ: {mean_improvement:+.4f} | ' +
              f'{above_diagonal}/{len(common_latents)} improved ({100*above_diagonal/len(common_latents):.1f}%)',
              fontsize=13)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Add legend
    plt.legend()
    
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
        'mean_improvement': float(mean_improvement),
        'median_improvement': float(median_improvement),
        'n_improved': above_diagonal,
        'n_worse': below_diagonal,
        'n_same': on_diagonal,
        'pct_improved': float(100 * above_diagonal / len(common_latents))
    }


def main():
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python plot_baseline_vs_best.py <baseline_results.json> <magic_planet_results.json> [output_path]")
        sys.exit(1)
    
    baseline_file = Path(sys.argv[1])
    magic_planet_file = Path(sys.argv[2])
    
    # Determine output path
    if len(sys.argv) >= 4:
        output_path = Path(sys.argv[3])
    else:
        output_path = magic_planet_file.parent / 'baseline_vs_best_f1.png'
    
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
    
    # Create scatter plot
    print("\nCreating scatter plot...")
    stats = create_scatter_plot(baseline_f1, best_f1, output_path)
    
    # Save statistics
    stats_path = output_path.parent / 'baseline_vs_best_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {stats_path}")


if __name__ == '__main__':
    main()

