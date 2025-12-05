#!/usr/bin/env python3
"""
Score generated labels using the detection scorer with vLLM.

This script efficiently scores all labels in a generated_labels JSON file
by batching across latents and computing summary statistics.

CRITICAL: vLLM must be initialized BEFORE loading any PyTorch datasets
to avoid CUDA deadlock. See VLLM_CUDA_FIX.md for details.
"""

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm
from transformers import AutoTokenizer

# Import vLLM client and scorer FIRST
from delphi.clients import Offline
from delphi.config import ConstructorConfig, SamplerConfig
from delphi.latents import LatentDataset, Latent
from delphi.scorers import DetectionScorer


def load_labels(labels_file: Path) -> tuple[Dict, List]:
    """
    Load labels from JSON file and group by latent_index.
    
    Returns:
        metadata: Metadata from the JSON file
        labels_by_latent: List of (latent_index, labels_for_that_latent)
    """
    print(f"Loading labels from {labels_file}...")
    with open(labels_file, "r") as f:
        data = json.load(f)
    
    metadata = data["metadata"]
    generated_labels = data["generated_labels"]
    
    # Group labels by latent_index
    labels_dict = defaultdict(list)
    for label_entry in generated_labels:
        labels_dict[label_entry["latent_index"]].append(label_entry)
    
    # Convert to sorted list of tuples for deterministic ordering
    labels_by_latent = sorted(labels_dict.items())
    
    print(f"Loaded {len(generated_labels)} labels for {len(labels_by_latent)} unique latents")
    print(f"Scale values: {metadata['scale_values']}")
    
    return metadata, labels_by_latent


async def score_labels_batch(
    scorer: DetectionScorer,
    records_dict: Dict[int, Any],
    labels_batch: List[tuple[int, List[Dict]]],
) -> List[Dict]:
    """
    Score a batch of labels across multiple latents.
    
    Args:
        scorer: DetectionScorer instance
        records_dict: Dictionary mapping latent_index to LatentRecord
        labels_batch: List of (latent_index, labels) tuples
    
    Returns:
        List of score results
    """
    results = []
    
    # Process each latent and its labels
    for latent_idx, labels in labels_batch:
        record = records_dict.get(latent_idx)
        
        if record is None:
            print(f"  Warning: Could not find LatentRecord for latent {latent_idx}, skipping...")
            continue
        
        # Score each label for this latent
        for label_entry in labels:
            # Set the explanation to the label text
            record.explanation = label_entry["label"]
            
            # Score using the detection scorer
            try:
                scorer_result = await scorer(record)
                
                # Extract results from ClassifierOutput objects
                score_data = scorer_result.score
                
                # Compute accuracy and other metrics
                correct_predictions = [s.correct for s in score_data if s.correct is not None]
                probabilities = [s.probability for s in score_data if s.probability is not None]
                
                accuracy = sum(correct_predictions) / len(correct_predictions) if correct_predictions else 0.0
                avg_probability = sum(probabilities) / len(probabilities) if probabilities else 0.0
                
                # Store detailed results
                result = {
                    "latent_index": latent_idx,
                    "label": label_entry["label"],
                    "scale": label_entry["scale"],
                    "label_index": label_entry.get("label_index", 0),
                    "accuracy": accuracy,
                    "num_correct": sum(correct_predictions),
                    "num_total": len(correct_predictions),
                    "avg_probability": avg_probability,
                    "per_example_results": [
                        {
                            "activating": s.activating,
                            "prediction": s.prediction,
                            "probability": s.probability,
                            "correct": s.correct,
                        }
                        for s in score_data
                    ]
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error scoring latent {latent_idx} with label '{label_entry['label'][:50]}...': {e}")
                continue
    
    return results


def compute_summary_statistics(all_results: List[Dict], metadata: Dict) -> Dict:
    """
    Compute summary statistics overall and by scale parameter.
    
    Args:
        all_results: List of all scoring results
        metadata: Metadata from the labels file
    
    Returns:
        Dictionary of summary statistics
    """
    if not all_results:
        return {
            "overall": {},
            "by_scale": {}
        }
    
    # Overall statistics
    overall_accuracies = [r["accuracy"] for r in all_results]
    overall_probabilities = [r["avg_probability"] for r in all_results]
    
    overall_stats = {
        "avg_accuracy": sum(overall_accuracies) / len(overall_accuracies),
        "avg_probability": sum(overall_probabilities) / len(overall_probabilities),
        "total_labels": len(all_results),
        "min_accuracy": min(overall_accuracies),
        "max_accuracy": max(overall_accuracies),
    }
    
    # Statistics by scale
    by_scale = defaultdict(list)
    for result in all_results:
        by_scale[result["scale"]].append(result)
    
    scale_stats = {}
    for scale, results in sorted(by_scale.items()):
        accuracies = [r["accuracy"] for r in results]
        probabilities = [r["avg_probability"] for r in results]
        
        scale_stats[str(scale)] = {
            "avg_accuracy": sum(accuracies) / len(accuracies),
            "avg_probability": sum(probabilities) / len(probabilities),
            "count": len(results),
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
        }
    
    return {
        "overall": overall_stats,
        "by_scale": scale_stats
    }


def load_latent_records(dataset: LatentDataset, latent_indices: set) -> Dict[int, Any]:
    """
    Load LatentRecords from dataset (must be called from sync context).
    
    Args:
        dataset: LatentDataset to load from
        latent_indices: Set of latent indices to load
    
    Returns:
        Dictionary mapping latent_index to LatentRecord
    """
    records_dict = {}
    print(f"Need to load {len(latent_indices)} unique latents")
    
    with tqdm(total=len(latent_indices), desc="Loading records") as pbar:
        for record in dataset:
            if record.latent.latent_index in latent_indices:
                records_dict[record.latent.latent_index] = record
                pbar.update(1)
                if len(records_dict) >= len(latent_indices):
                    break  # Found all records we need
    
    return records_dict


def display_summary(summary_stats: Dict):
    """Display summary statistics in a readable format."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    overall = summary_stats["overall"]
    print("\nOverall Statistics:")
    print(f"  Total labels scored: {overall.get('total_labels', 0)}")
    print(f"  Average accuracy: {overall.get('avg_accuracy', 0):.4f}")
    print(f"  Average probability: {overall.get('avg_probability', 0):.4f}")
    print(f"  Min accuracy: {overall.get('min_accuracy', 0):.4f}")
    print(f"  Max accuracy: {overall.get('max_accuracy', 0):.4f}")
    
    print("\nStatistics by Scale:")
    by_scale = summary_stats["by_scale"]
    print(f"{'Scale':<10} {'Count':<10} {'Avg Accuracy':<15} {'Avg Probability':<15} {'Min Acc':<10} {'Max Acc':<10}")
    print("-" * 80)
    for scale in sorted(by_scale.keys(), key=float):
        stats = by_scale[scale]
        print(f"{scale:<10} {stats['count']:<10} {stats['avg_accuracy']:<15.4f} "
              f"{stats['avg_probability']:<15.4f} {stats['min_accuracy']:<10.4f} {stats['max_accuracy']:<10.4f}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Score generated labels using detection scorer"
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        required=True,
        help="Path to generated_labels JSON file"
    )
    parser.add_argument(
        "--activations-dir",
        type=Path,
        required=True,
        help="Path to activations directory (e.g., neuronpedia_activations/19-llamascope-res-32k)"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model to use for scoring (default: meta-llama/Meta-Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer to use (defaults to same as --model)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="Number of GPUs to use (default: 2)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length (default: 4096)"
    )
    parser.add_argument(
        "--num-examples-per-prompt",
        type=int,
        default=5,
        help="Number of examples per scorer prompt (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of latents to process in parallel (default: 10)"
    )
    parser.add_argument(
        "--max-latents",
        type=int,
        default=None,
        help="Maximum number of latents to score (for testing)"
    )
    
    args = parser.parse_args()
    
    # Use model as tokenizer if not specified
    if args.tokenizer is None:
        args.tokenizer = args.model
    
    print("=" * 80)
    print("Label Scoring Script")
    print("=" * 80)
    print(f"Labels file: {args.labels_file}")
    print(f"Activations directory: {args.activations_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Model: {args.model}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Max model length: {args.max_model_len}")
    print(f"Examples per prompt: {args.num_examples_per_prompt}")
    print(f"Batch size: {args.batch_size}")
    
    # Step 1: Load labels from JSON
    metadata, labels_by_latent = load_labels(args.labels_file)
    
    if args.max_latents:
        labels_by_latent = labels_by_latent[:args.max_latents]
        print(f"\nLimiting to first {args.max_latents} latents for testing")
    
    # Step 2: Initialize vLLM client FIRST (before dataset loading)
    print("\n" + "=" * 80)
    print("STEP 1: Initializing vLLM client (BEFORE dataset loading)")
    print("=" * 80)
    
    llm_client = Offline(
        args.model,
        max_memory=0.85,
        max_model_len=args.max_model_len,
        num_gpus=args.num_gpus,
        statistics=False,
    )
    print("✅ vLLM client initialized")
    
    # Step 3: Create DetectionScorer
    scorer = DetectionScorer(
        llm_client,
        n_examples_shown=args.num_examples_per_prompt,
        verbose=False,
        log_prob=True,
    )
    print("✅ DetectionScorer created")
    
    # Step 4: Load tokenizer and dataset (AFTER vLLM initialization)
    print("\n" + "=" * 80)
    print("STEP 2: Loading dataset (AFTER vLLM initialization)")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"✅ Tokenizer loaded: {args.tokenizer}")
    
    # Configure sampler and constructor
    sampler_cfg = SamplerConfig(
        n_examples_train=40,
        n_examples_test=50,
        n_quantiles=10,
        train_type="quantiles",
        test_type="quantiles",
    )
    
    constructor_cfg = ConstructorConfig(
        min_examples=50,  # Lower threshold for neuronpedia data
        example_ctx_len=32,
        n_non_activating=50,
        non_activating_source="random",
    )
    
    # Determine module name from activations directory
    # The directory structure should be: activations_dir/config.json
    config_path = args.activations_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        # Extract layer from SAE name (e.g., "19-llamascope-res-32k" -> layer 19)
        sae_name = config.get("sae_name", "")
        if sae_name:
            # Parse layer number from SAE name
            layer_num = sae_name.split("-")[0]
            module_name = f"model.layers.{layer_num}"
        else:
            module_name = "model.layers.19"  # Default
    else:
        module_name = "model.layers.19"
    
    print(f"Using module name: {module_name}")
    
    # LatentDataset expects the structure: raw_dir/module_name/*.safetensors
    # But neuronpedia data is in: activations_dir/*.safetensors
    # So we need to use the parent as raw_dir and the dir name as the module
    # However, the module name needs to match what we use for lookups
    # We'll use the activations_dir name as the module to match the filesystem
    
    # Create LatentDataset
    dataset = LatentDataset(
        raw_dir=args.activations_dir.parent,  # Parent dir (neuronpedia_activations)
        modules=[args.activations_dir.name],  # SAE directory name (19-llamascope-res-32k)
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
        tokenizer=tokenizer,
    )
    print("✅ LatentDataset loaded")
    
    # Step 5: Load all needed LatentRecords into memory
    print("\n" + "=" * 80)
    print("STEP 3: Loading LatentRecords from dataset")
    print("=" * 80)
    
    # Collect all latent indices we need
    all_latent_indices = {latent_idx for latent_idx, _ in labels_by_latent}
    
    # Load all records (dataset is an iterator, can only iterate once)
    # Must be done in synchronous context to avoid event loop conflicts
    records_dict = load_latent_records(dataset, all_latent_indices)
    
    print(f"✅ Loaded {len(records_dict)} LatentRecords")
    
    # Step 6: Score all labels in batches (async)
    print("\n" + "=" * 80)
    print("STEP 4: Scoring labels")
    print("=" * 80)
    
    # Run the async scoring
    all_results = asyncio.run(score_all_labels(
        scorer,
        records_dict,
        labels_by_latent,
        args.batch_size
    ))
    
    print(f"\n✅ Scored {len(all_results)} labels")
    
    # Step 7: Compute summary statistics
    print("\n" + "=" * 80)
    print("STEP 5: Computing summary statistics")
    print("=" * 80)
    
    summary_stats = compute_summary_statistics(all_results, metadata)
    
    # Step 8: Save output
    output_data = {
        "metadata": metadata,
        "scores": all_results,
        "summary_statistics": summary_stats
    }
    
    print(f"\nSaving results to {args.output_file}...")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Results saved to {args.output_file}")
    
    # Display summary
    display_summary(summary_stats)
    
    print("\n✅ COMPLETE!")


async def score_all_labels(
    scorer: DetectionScorer,
    records_dict: Dict[int, Any],
    labels_by_latent: List[tuple[int, List[Dict]]],
    batch_size: int
) -> List[Dict]:
    """
    Async function to score all labels in batches.
    
    Args:
        scorer: DetectionScorer instance
        records_dict: Dictionary of LatentRecords
        labels_by_latent: List of (latent_index, labels) tuples
        batch_size: Batch size for processing
    
    Returns:
        List of all results
    """
    all_results = []
    
    with tqdm(total=len(labels_by_latent), desc="Scoring latents") as pbar:
        for i in range(0, len(labels_by_latent), batch_size):
            batch = labels_by_latent[i:i + batch_size]
            
            # Score this batch
            batch_results = await score_labels_batch(
                scorer,
                records_dict,
                batch,
            )
            
            all_results.extend(batch_results)
            pbar.update(len(batch))
    
    return all_results


if __name__ == "__main__":
    main()

