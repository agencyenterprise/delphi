#!/usr/bin/env python3
"""
Evaluate SAE labels using activation data and detection scoring.

This script loads labels from JSON files (with or without scale values),
loads corresponding activation data, and scores each label using the
detection scorer. Results are saved to a file and summary statistics
are printed.

Usage:
    python eval_labels.py --labels labels.json --activations neuronpedia_activations --layer 19-llamascope-res-32k
"""

import argparse
import asyncio
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer

from delphi.clients.offline import Offline
from delphi.config import ConstructorConfig, SamplerConfig
from delphi.latents import LatentDataset, LatentRecord
from delphi.scorers.classifier.detection import DetectionScorer


@dataclass
class LabelToEvaluate:
    """A single label to evaluate."""
    latent_index: int
    label: str
    scale: Optional[float] = None
    label_index: Optional[int] = None


@dataclass
class EvaluationResult:
    """Result of evaluating a single label."""
    latent_index: int
    label: str
    scale: Optional[float]
    label_index: Optional[int]
    accuracy: Optional[float]  # None indicates error
    num_correct: int
    num_total: int
    num_activating: int
    num_non_activating: int
    activating_accuracy: Optional[float]  # None indicates error
    non_activating_accuracy: Optional[float]  # None indicates error
    error: Optional[str] = None  # Error message if evaluation failed


def load_labels(labels_path: Path) -> tuple[dict, list[LabelToEvaluate]]:
    """
    Load labels from JSON file.
    
    Args:
        labels_path: Path to labels JSON file
    
    Returns:
        Tuple of (metadata, list of labels to evaluate)
    """
    with open(labels_path, 'r') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    generated_labels = data.get('generated_labels', [])
    
    labels = []
    for item in generated_labels:
        labels.append(LabelToEvaluate(
            latent_index=item['latent_index'],
            label=item['label'],
            scale=item.get('scale'),
            label_index=item.get('label_index')
        ))
    
    return metadata, labels


async def evaluate_single_label(
    label: LabelToEvaluate,
    latent_records_dict: dict[int, LatentRecord],
    scorer: DetectionScorer,
) -> EvaluationResult:
    """
    Evaluate a single label using detection scoring.
    
    Args:
        label: The label to evaluate
        latent_records_dict: Pre-loaded dictionary mapping latent_index -> LatentRecord
        scorer: Detection scorer instance
    
    Returns:
        EvaluationResult with scoring metrics
    """
    try:
        # Look up the pre-loaded record
        if label.latent_index not in latent_records_dict:
            raise ValueError(f"Latent {label.latent_index} not found (may have been filtered due to insufficient examples)")
        
        record = latent_records_dict[label.latent_index]
        
        # Set the explanation to our label
        record.explanation = label.label
        
        # Run the scorer
        scorer_result = await scorer(record)
        
        # Calculate metrics
        results = scorer_result.score
        num_correct = sum(1 for r in results if r.correct)
        num_total = len(results)
        accuracy = num_correct / num_total if num_total > 0 else 0.0
        
        # Separate activating vs non-activating
        activating_results = [r for r in results if r.activating]
        non_activating_results = [r for r in results if not r.activating]
        
        num_activating = len(activating_results)
        num_non_activating = len(non_activating_results)
        
        activating_correct = sum(1 for r in activating_results if r.correct)
        non_activating_correct = sum(1 for r in non_activating_results if r.correct)
        
        activating_accuracy = activating_correct / num_activating if num_activating > 0 else 0.0
        non_activating_accuracy = non_activating_correct / num_non_activating if num_non_activating > 0 else 0.0
        
        return EvaluationResult(
            latent_index=label.latent_index,
            label=label.label,
            scale=label.scale,
            label_index=label.label_index,
            accuracy=accuracy,
            num_correct=num_correct,
            num_total=num_total,
            num_activating=num_activating,
            num_non_activating=num_non_activating,
            activating_accuracy=activating_accuracy,
            non_activating_accuracy=non_activating_accuracy,
        )
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n⚠ Error evaluating latent {label.latent_index}: {error_msg}")
        # Return an error result with None for all metrics
        return EvaluationResult(
            latent_index=label.latent_index,
            label=label.label,
            scale=label.scale,
            label_index=label.label_index,
            accuracy=None,
            num_correct=0,
            num_total=0,
            num_activating=0,
            num_non_activating=0,
            activating_accuracy=None,
            non_activating_accuracy=None,
            error=error_msg,
        )


async def evaluate_labels(
    labels: list[LabelToEvaluate],
    activations_dir: Path,
    layer: str,
    model: str,
    num_gpus: int,
    max_model_len: int,
) -> list[EvaluationResult]:
    """
    Evaluate all labels in parallel using vLLM batching.
    
    Args:
        labels: List of labels to evaluate
        activations_dir: Directory containing activation data
        layer: Layer identifier
        model: Model name for LLM
        num_gpus: Number of GPUs to use
        max_model_len: Maximum model context length
    
    Returns:
        List of evaluation results
    """
    print("="*80)
    print("INITIALIZING")
    print("="*80)
    
    # Load tokenizer
    print(f"Loading tokenizer: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    # Initialize vLLM client
    print(f"Initializing vLLM with {num_gpus} GPUs...")
    client = Offline(
        model=model,
        num_gpus=num_gpus,
        max_model_len=max_model_len,
        batch_size=100,
    )
    
    # Initialize scorer
    scorer = DetectionScorer(
        client=client,
        n_examples_shown=5,  # Number of examples to show per prompt
        verbose=False,
        log_prob=False,
        temperature=0.0,
    )
    
    # Load configuration from the activation data
    print(f"Activation data directory: {activations_dir / layer}")
    
    # Read ctx_len from the config.json in the activation directory
    config_path = activations_dir / layer / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        activation_config = json.load(f)
    
    ctx_len = activation_config.get("ctx_len")
    if ctx_len is None:
        raise ValueError(f"ctx_len not found in config: {config_path}")
    
    print(f"  Context length from activation data: {ctx_len}")
    
    sampler_cfg = SamplerConfig(
        n_examples_train=40,
        n_examples_test=50,
        n_quantiles=5,
        train_type="quantiles",
        test_type="quantiles",
    )
    constructor_cfg = ConstructorConfig(
        example_ctx_len=ctx_len,  # Read from activation config
        min_examples=50,
        n_non_activating=50,
        center_examples=True,
        non_activating_source="random",
    )
    
    print(f"✓ Configuration prepared")
    
    # Load ALL latent records at once (much more efficient!)
    print(f"\nLoading activation data for {len(labels)} latents...")
    unique_latent_indices = list(set(label.latent_index for label in labels))
    print(f"  Unique latents to load: {len(unique_latent_indices)}")
    
    latents_tensor = torch.tensor(unique_latent_indices, dtype=torch.int64)
    latents_dict = {layer: latents_tensor}
    
    dataset = LatentDataset(
        raw_dir=activations_dir,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
        tokenizer=tokenizer,
        modules=[layer],
        latents=latents_dict,
    )
    
    # Build a dictionary mapping latent_index -> LatentRecord
    print(f"  Building latent records dictionary...")
    latent_records_dict = {}
    filtered_count = 0
    async for latent_record in dataset:
        latent_records_dict[latent_record.latent.latent_index] = latent_record
    
    filtered_count = len(unique_latent_indices) - len(latent_records_dict)
    if filtered_count > 0:
        print(f"  ⚠ {filtered_count} latents filtered out (insufficient examples)")
    print(f"✓ Loaded {len(latent_records_dict)} latent records")
    
    # Evaluate all labels concurrently
    # vLLM will automatically batch the requests
    print("\n" + "="*80)
    print("EVALUATING LABELS")
    print("="*80)
    print(f"Total labels to evaluate: {len(labels)}")
    print("Note: vLLM will automatically batch requests for efficiency")
    print()
    
    # Create all tasks at once - vLLM handles batching
    tasks = [
        evaluate_single_label(
            label,
            latent_records_dict,
            scorer,
        )
        for label in labels
    ]
    
    # Run all tasks concurrently with progress bar
    results = await tqdm_asyncio.gather(
        *tasks,
        desc="Evaluating labels",
        total=len(tasks),
    )
    
    return results


def print_statistics(results: list[EvaluationResult], metadata: dict):
    """
    Print summary statistics for evaluation results.
    
    Args:
        results: List of evaluation results
        metadata: Metadata from the labels file
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Separate successful results from errors
    successful_results = [r for r in results if r.accuracy is not None]
    error_results = [r for r in results if r.accuracy is None]
    
    if error_results:
        print(f"\n⚠ WARNING: {len(error_results)} labels failed to evaluate (marked as errors)")
        print(f"  Successful evaluations: {len(successful_results)}/{len(results)}")
    
    if not successful_results:
        print("\n⚠ No successful evaluations to report statistics")
        return
    
    # Overall statistics (only for successful results)
    # Filter out None values (shouldn't happen for successful_results, but be safe)
    all_accuracies = [r.accuracy for r in successful_results if r.accuracy is not None]
    all_activating_accuracies = [r.activating_accuracy for r in successful_results if r.activating_accuracy is not None and r.num_activating > 0]
    all_non_activating_accuracies = [r.non_activating_accuracy for r in successful_results if r.non_activating_accuracy is not None and r.num_non_activating > 0]
    
    print("\nOVERALL (successful evaluations only):")
    print(f"  Total labels evaluated: {len(successful_results)}")
    print(f"  Mean accuracy: {np.mean(all_accuracies):.3f}")
    print(f"  Std accuracy: {np.std(all_accuracies):.3f}")
    print(f"  Median accuracy: {np.median(all_accuracies):.3f}")
    print(f"  Min accuracy: {np.min(all_accuracies):.3f}")
    print(f"  Max accuracy: {np.max(all_accuracies):.3f}")
    
    if all_activating_accuracies:
        print(f"\n  Mean activating accuracy: {np.mean(all_activating_accuracies):.3f}")
    if all_non_activating_accuracies:
        print(f"  Mean non-activating accuracy: {np.mean(all_non_activating_accuracies):.3f}")
    
    # Per-scale statistics (if applicable, only for successful results)
    scale_results = defaultdict(list)
    for result in successful_results:
        if result.scale is not None:
            scale_results[result.scale].append(result)
    
    if scale_results:
        print("\nPER-SCALE:")
        for scale in sorted(scale_results.keys()):
            scale_accs = [r.accuracy for r in scale_results[scale] if r.accuracy is not None]
            if scale_accs:
                print(f"\n  Scale {scale}:")
                print(f"    N: {len(scale_accs)}")
                print(f"    Mean accuracy: {np.mean(scale_accs):.3f}")
                print(f"    Std accuracy: {np.std(scale_accs):.3f}")
                print(f"    Median accuracy: {np.median(scale_accs):.3f}")
    
    # Top and bottom performers (only successful results with non-None accuracy)
    sortable_results = [r for r in successful_results if r.accuracy is not None]
    sorted_results = sorted(sortable_results, key=lambda r: r.accuracy or 0.0, reverse=True)
    
    if len(sorted_results) > 0:
        print("\nTOP 5 BEST LABELS:")
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"  {i}. Latent {result.latent_index}: {result.accuracy:.3f} - \"{result.label[:60]}...\"")
        
        print("\nTOP 5 WORST LABELS:")
        for i, result in enumerate(sorted_results[-5:], 1):
            print(f"  {i}. Latent {result.latent_index}: {result.accuracy:.3f} - \"{result.label[:60]}...\"")
    
    if error_results:
        print(f"\nERRORED LABELS ({len(error_results)}):")
        for i, result in enumerate(error_results[:10], 1):  # Show first 10 errors
            print(f"  {i}. Latent {result.latent_index}: ERROR - {result.error[:80] if result.error else 'Unknown error'}")


def save_results(
    results: list[EvaluationResult],
    metadata: dict,
    output_path: Path,
):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: List of evaluation results
        metadata: Original metadata from labels file
        output_path: Path to save results
    """
    # Separate successful from error results
    successful_results = [r for r in results if r.accuracy is not None]
    error_results = [r for r in results if r.accuracy is None]
    
    # Get accuracies from successful results (filter out any Nones)
    successful_accuracies = [r.accuracy for r in successful_results if r.accuracy is not None]
    
    output_data = {
        "metadata": metadata,
        "evaluation_results": [asdict(r) for r in results],
        "summary": {
            "total_labels": len(results),
            "successful_evaluations": len(successful_results),
            "failed_evaluations": len(error_results),
            "mean_accuracy": float(np.mean(successful_accuracies)) if successful_accuracies else None,
            "std_accuracy": float(np.std(successful_accuracies)) if successful_accuracies else None,
            "median_accuracy": float(np.median(successful_accuracies)) if successful_accuracies else None,
        }
    }
    
    # Add per-scale summary if applicable (only successful results)
    scale_results = defaultdict(list)
    for result in successful_results:
        if result.scale is not None:
            scale_results[result.scale].append(result)
    
    if scale_results:
        output_data["per_scale_summary"] = {}
        for scale in sorted(scale_results.keys()):
            scale_accs = [r.accuracy for r in scale_results[scale] if r.accuracy is not None]
            if scale_accs:  # Only add if we have valid accuracies
                output_data["per_scale_summary"][str(scale)] = {
                    "n": len(scale_accs),
                    "mean_accuracy": float(np.mean(scale_accs)),
                    "std_accuracy": float(np.std(scale_accs)),
                    "median_accuracy": float(np.median(scale_accs)),
                }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SAE labels using detection scoring"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to labels JSON file"
    )
    parser.add_argument(
        "--activations",
        type=Path,
        default=Path("neuronpedia_activations"),
        help="Directory containing activation data (default: neuronpedia_activations)"
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="19-llamascope-res-32k",
        help="Layer identifier (default: 19-llamascope-res-32k)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model for evaluation (default: meta-llama/Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length (default: 4096)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for results (default: auto-generated based on input)"
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=None,
        help="Maximum number of labels to evaluate (for testing, default: all)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("SAE LABEL EVALUATION")
    print("="*80)
    print(f"Labels file: {args.labels}")
    print(f"Activations dir: {args.activations}")
    print(f"Layer: {args.layer}")
    print(f"Model: {args.model}")
    print(f"GPUs: {args.num_gpus}")
    
    # Load labels
    print("\nLoading labels...")
    metadata, labels = load_labels(args.labels)
    print(f"✓ Loaded {len(labels)} labels")
    
    if metadata:
        print(f"  Dataset: {metadata.get('dataset_name', 'unknown')}")
        print(f"  Layer: {metadata.get('layer', 'unknown')}")
        if 'scale_values' in metadata:
            print(f"  Scale values: {metadata['scale_values']}")
    
    # Limit labels if requested
    if args.max_labels:
        labels = labels[:args.max_labels]
        print(f"  Limited to first {len(labels)} labels for testing")
    
    # Evaluate labels
    results = await evaluate_labels(
        labels,
        args.activations,
        args.layer,
        args.model,
        args.num_gpus,
        args.max_model_len,
    )
    
    # Print statistics
    print_statistics(results, metadata)
    
    # Save results
    if args.output is None:
        # Auto-generate output filename
        input_stem = args.labels.stem
        args.output = args.labels.parent / f"{input_stem}_eval_results.json"
    
    save_results(results, metadata, args.output)
    
    print("\n" + "="*80)
    print("✓ EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())

