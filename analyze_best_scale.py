#!/usr/bin/env python3
"""
Analyze detailed_results.json by finding the best scale for each latent.

For each latent index:
- Find the label with the highest F1 score (across all scales)
- Find the label with the highest accuracy (across all scales)

Then compute summary statistics for these "best" entries.
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import sys


def load_results(filepath):
    """Load the detailed results JSON file."""
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} total entries")
    return data


def group_by_latent(data):
    """Group entries by latent_index."""
    grouped = defaultdict(list)
    for entry in data:
        latent_idx = entry['latent_index']
        grouped[latent_idx].append(entry)
    return grouped


def find_best_entries(grouped_data, metric_name):
    """
    For each latent, find the entry with the highest value for the given metric.
    
    Args:
        grouped_data: Dict mapping latent_index -> list of entries
        metric_name: Either 'f1' or 'accuracy'
    
    Returns:
        List of best entries (one per latent)
    """
    best_entries = []
    
    for latent_idx, entries in grouped_data.items():
        # Find the entry with the highest metric value
        best_entry = max(entries, key=lambda x: x['metrics'][metric_name])
        best_entries.append(best_entry)
    
    return best_entries


def compute_summary_statistics(entries, metric_name):
    """
    Compute summary statistics for the given metric across all entries.
    
    Args:
        entries: List of entries (each with a 'metrics' field)
        metric_name: The metric to summarize
    
    Returns:
        Dict with summary statistics
    """
    values = [entry['metrics'][metric_name] for entry in entries]
    
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75)),
        'count': len(values)
    }


def compute_all_metrics_summary(entries):
    """Compute summary statistics for all metrics."""
    all_metrics = ['accuracy', 'precision', 'recall', 'f1', 'error_rate']
    
    summary = {}
    for metric in all_metrics:
        summary[metric] = compute_summary_statistics(entries, metric)
    
    return summary


def analyze_scale_distribution(entries):
    """Analyze which scales were selected as 'best'."""
    scales = [entry['scale'] for entry in entries]
    scale_counts = defaultdict(int)
    
    for scale in scales:
        scale_counts[scale] += 1
    
    return {
        'scale_distribution': dict(sorted(scale_counts.items())),
        'total_latents': len(entries)
    }


def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python analyze_best_scale.py <path_to_detailed_results.json> [output_dir]")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    
    # Determine output directory
    if len(sys.argv) >= 3:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = input_file.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_results(input_file)
    
    # Group by latent index
    print("\nGrouping by latent index...")
    grouped_data = group_by_latent(data)
    print(f"Found {len(grouped_data)} unique latent indices")
    
    # Find best entries for F1 score
    print("\nFinding best scale per latent (by F1 score)...")
    best_f1_entries = find_best_entries(grouped_data, 'f1')
    print(f"Selected {len(best_f1_entries)} best entries (by F1)")
    
    # Find best entries for accuracy
    print("\nFinding best scale per latent (by accuracy)...")
    best_acc_entries = find_best_entries(grouped_data, 'accuracy')
    print(f"Selected {len(best_acc_entries)} best entries (by accuracy)")
    
    # Compute summary statistics for F1-optimized selection
    print("\nComputing summary statistics for F1-optimized selection...")
    f1_summary = compute_all_metrics_summary(best_f1_entries)
    f1_scale_dist = analyze_scale_distribution(best_f1_entries)
    
    # Compute summary statistics for accuracy-optimized selection
    print("\nComputing summary statistics for accuracy-optimized selection...")
    acc_summary = compute_all_metrics_summary(best_acc_entries)
    acc_scale_dist = analyze_scale_distribution(best_acc_entries)
    
    # Prepare output
    output = {
        'description': 'Best scale per latent analysis',
        'input_file': str(input_file),
        'total_entries': len(data),
        'unique_latents': len(grouped_data),
        'f1_optimized': {
            'description': 'Statistics when selecting the scale with highest F1 for each latent',
            'summary_statistics': f1_summary,
            'scale_distribution': f1_scale_dist
        },
        'accuracy_optimized': {
            'description': 'Statistics when selecting the scale with highest accuracy for each latent',
            'summary_statistics': acc_summary,
            'scale_distribution': acc_scale_dist
        }
    }
    
    # Save results
    output_file = output_dir / 'best_scale_analysis.json'
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary to console
    print("\n" + "="*80)
    print("SUMMARY - F1 OPTIMIZED (selecting best F1 scale per latent)")
    print("="*80)
    print(f"Number of latents: {len(best_f1_entries)}")
    print(f"\nF1 Score:")
    print(f"  Mean:   {f1_summary['f1']['mean']:.4f}")
    print(f"  Std:    {f1_summary['f1']['std']:.4f}")
    print(f"  Median: {f1_summary['f1']['median']:.4f}")
    print(f"  Min:    {f1_summary['f1']['min']:.4f}")
    print(f"  Max:    {f1_summary['f1']['max']:.4f}")
    print(f"\nAccuracy:")
    print(f"  Mean:   {f1_summary['accuracy']['mean']:.4f}")
    print(f"  Std:    {f1_summary['accuracy']['std']:.4f}")
    print(f"  Median: {f1_summary['accuracy']['median']:.4f}")
    print(f"\nScale Distribution:")
    for scale, count in sorted(f1_scale_dist['scale_distribution'].items()):
        pct = 100 * count / f1_scale_dist['total_latents']
        print(f"  {scale}: {count} ({pct:.1f}%)")
    
    print("\n" + "="*80)
    print("SUMMARY - ACCURACY OPTIMIZED (selecting best accuracy scale per latent)")
    print("="*80)
    print(f"Number of latents: {len(best_acc_entries)}")
    print(f"\nAccuracy:")
    print(f"  Mean:   {acc_summary['accuracy']['mean']:.4f}")
    print(f"  Std:    {acc_summary['accuracy']['std']:.4f}")
    print(f"  Median: {acc_summary['accuracy']['median']:.4f}")
    print(f"  Min:    {acc_summary['accuracy']['min']:.4f}")
    print(f"  Max:    {acc_summary['accuracy']['max']:.4f}")
    print(f"\nF1 Score:")
    print(f"  Mean:   {acc_summary['f1']['mean']:.4f}")
    print(f"  Std:    {acc_summary['f1']['std']:.4f}")
    print(f"  Median: {acc_summary['f1']['median']:.4f}")
    print(f"\nScale Distribution:")
    for scale, count in sorted(acc_scale_dist['scale_distribution'].items()):
        pct = 100 * count / acc_scale_dist['total_latents']
        print(f"  {scale}: {count} ({pct:.1f}%)")
    
    print("\n" + "="*80)
    print(f"Full results saved to: {output_file}")
    print("="*80)


if __name__ == '__main__':
    main()

