#!/usr/bin/env python3
"""
Score existing auto-interp labels using delphi scorers.

This script loads pre-generated labels (either baseline or experimental with multiple scales)
and scores them using a delphi scorer (e.g., DetectionScorer). It efficiently pipelines the
CPU-bound LatentRecord construction with GPU-bound vLLM inference, with separate progress
bars for visibility.

Usage:
    python score_existing_labels.py \\
        --labels labels_baseline.json \\
        --activations neuronpedia_activations \\
        --module 19-llamascope-res-32k \\
        --scorer detection \\
        --output results/baseline_detection
"""

import argparse
import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from delphi.clients import Offline
from delphi.config import ConstructorConfig, SamplerConfig
from delphi.latents import LatentDataset, LatentRecord
from delphi.scorers import DetectionScorer
from delphi.scorers.classifier.sample import ClassifierOutput


@dataclass
class LabelInfo:
    """Metadata about a label to be scored."""
    latent_index: int
    label: str
    scale: Optional[float] = None
    label_index: Optional[int] = None  # For multi-scale experiments
    

@dataclass
class ScoringTask:
    """A latent record paired with label metadata."""
    record: LatentRecord
    label_info: LabelInfo


def load_labels(label_file: Path) -> list[LabelInfo]:
    """
    Load labels from JSON file.
    
    Handles both formats:
    - Baseline: {"generated_labels": [{"latent_index": 21, "label": "..."}]}
    - Experimental: {"generated_labels": [{"latent_index": 21, "label": "...", "scale": 2.0, "label_index": 0}]}
    """
    with open(label_file) as f:
        data = json.load(f)
    
    labels = []
    for entry in data["generated_labels"]:
        labels.append(LabelInfo(
            latent_index=entry["latent_index"],
            label=entry["label"],
            scale=entry.get("scale"),
            label_index=entry.get("label_index")
        ))
    
    print(f"✓ Loaded {len(labels)} labels from {label_file}")
    
    # Print summary
    unique_latents = len(set(l.latent_index for l in labels))
    print(f"  - {unique_latents} unique latents")
    if labels[0].scale is not None:
        scales = set(l.scale for l in labels)
        print(f"  - {len(scales)} scale values: {sorted(scales)}")
    
    return labels


def create_latent_dataset(
    activations_dir: Path,
    module: str,
    latent_indices: list[int],
    tokenizer_name: str = "meta-llama/Llama-3.1-8B"
) -> LatentDataset:
    """
    Create a LatentDataset that only loads the specified latent indices.
    
    This is much more efficient than loading all latents when we only need a subset.
    """
    print(f"\n{'='*80}")
    print(f"Initializing LatentDataset")
    print(f"{'='*80}")
    print(f"Activations dir: {activations_dir}")
    print(f"Module: {module}")
    print(f"Loading {len(latent_indices)} latents...")
    
    # Read the cache config to get the context length
    config_path = activations_dir / module / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        cache_config = json.load(f)
    
    ctx_len = cache_config.get("ctx_len")
    if ctx_len is None:
        raise ValueError(f"ctx_len not found in config: {config_path}")
    
    print(f"Cache context length: {ctx_len}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Create dataset with only selected latents
    latents_dict = {module: torch.tensor(sorted(set(latent_indices)), dtype=torch.int64)}
    
    # Configure how examples are constructed and sampled
    # CRITICAL: example_ctx_len must match the cache context length
    constructor_cfg = ConstructorConfig(
        example_ctx_len=ctx_len,  # Read from cache config
        n_non_activating=10,  # Number of non-activating examples
        min_examples=3,  # Minimum examples needed (lower than default 200 for efficiency)
        non_activating_source="random",  # Use random non-activating examples
        center_examples=True,  # Center examples on the latent activation
    )
    
    sampler_cfg = SamplerConfig(
        n_examples_train=5,  # Training examples for explanation generation
        n_examples_test=10,  # Test examples for scoring
        n_quantiles=5,  # Number of quantile-based examples
        train_type="quantiles",  # Use quantile-based sampling
        test_type="quantiles",  # Use quantile-based sampling for test
    )
    
    import sys
    print("Creating LatentDataset...")
    sys.stdout.flush()
    
    dataset = LatentDataset(
        raw_dir=activations_dir,
        modules=[module],
        latents=latents_dict,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
        tokenizer=tokenizer,
    )
    
    print(f"✓ Dataset initialized")
    print(f"  Token tensor shape: {dataset.tokens.shape if dataset.tokens is not None else 'None'}")
    sys.stdout.flush()
    return dataset


class LabelScoringLoader:
    """
    Async loader that constructs LatentRecords and pairs them with label info.
    
    This handles the CPU-bound record construction in a pipelined way, allowing
    the GPU-bound scoring to proceed while records are still being built.
    """
    
    def __init__(
        self,
        dataset: LatentDataset,
        labels: list[LabelInfo],
    ):
        self.dataset = dataset
        self.labels = labels
        
        # Build mapping from latent_index to labels
        self.latent_to_labels: dict[int, list[LabelInfo]] = defaultdict(list)
        for label_info in labels:
            self.latent_to_labels[label_info.latent_index].append(label_info)
        
        self.total_tasks = len(labels)
        self.constructed = 0
    
    async def __aiter__(self) -> AsyncIterator[ScoringTask]:
        """
        Iterate through the dataset, yielding ScoringTasks.
        
        Uses the dataset's async iterator which already handles concurrent
        processing of latents in a thread pool via @asyncify.
        """
        async for record in self.dataset:
            if record is None:
                continue
            
            latent_idx = record.latent.latent_index
            
            # Yield a separate task for each label associated with this latent
            if latent_idx in self.latent_to_labels:
                for label_info in self.latent_to_labels[latent_idx]:
                    self.constructed += 1
                    yield ScoringTask(record=record, label_info=label_info)


async def score_task(
    task: ScoringTask,
    scorer: DetectionScorer,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """
    Score a single task (LatentRecord with a label).
    
    Returns a dictionary with scoring results and metadata.
    """
    async with semaphore:
        # Set the explanation to the label
        task.record.explanation = task.label_info.label
        
        # Ensure we have non-activating examples for the scorer
        task.record.extra_examples = task.record.not_active
        
        # Run the scorer
        result = await scorer(task.record)
        
        # Extract scores
        scores: list[ClassifierOutput] = result.score
        
        # Compute metrics
        predictions = [s.prediction for s in scores if s.prediction is not None]
        actuals = [s.activating for s in scores]
        correct = [s.correct for s in scores if s.correct is not None]
        
        tp = sum(1 for s in scores if s.activating and s.prediction)
        fp = sum(1 for s in scores if not s.activating and s.prediction)
        tn = sum(1 for s in scores if not s.activating and not s.prediction)
        fn = sum(1 for s in scores if s.activating and not s.prediction)
        
        accuracy = sum(correct) / len(correct) if correct else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Build result dictionary
        return {
            "latent_index": task.label_info.latent_index,
            "label": task.label_info.label,
            "scale": task.label_info.scale,
            "label_index": task.label_info.label_index,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            },
            "per_example_scores": [
                {
                    "activating": s.activating,
                    "prediction": s.prediction,
                    "probability": s.probability,
                    "correct": s.correct,
                    "distance": s.distance,
                }
                for s in scores
            ],
            "latent_metadata": {
                "per_token_frequency": task.record.per_token_frequency,
                "per_context_frequency": task.record.per_context_frequency,
                "max_activation": task.record.max_activation,
                "n_train_examples": len(task.record.train),
                "n_test_examples": len(task.record.test),
                "n_not_active": len(task.record.not_active),
            }
        }


async def run_scoring_pipeline(
    loader: LabelScoringLoader,
    scorer: DetectionScorer,
    max_concurrent: int = 10,
) -> list[dict[str, Any]]:
    """
    Run the scoring pipeline with simple progress tracking.
    """
    import sys
    import time
    
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = set()
    
    constructed = 0
    scored = 0
    update_lock = asyncio.Lock()
    start_time = time.time()
    
    async def process_and_update(task: ScoringTask):
        nonlocal scored
        result = await score_task(task, scorer, semaphore)
        
        async with update_lock:
            scored += 1
            elapsed = time.time() - start_time
            rate = scored / elapsed if elapsed > 0 else 0
            
            # Print EVERY completion for debugging
            print(
                f"[SCORING] {scored}/{loader.total_tasks} "
                f"({scored*100//loader.total_tasks}%) | "
                f"Latent {result['latent_index']} | "
                f"F1: {result['metrics']['f1']:.3f} | "
                f"Rate: {rate:.1f}/s | "
                f"Elapsed: {elapsed:.0f}s",
                file=sys.stderr
            )
            sys.stderr.flush()
        return result
    
    print(f"\n[PIPELINE] Starting scoring pipeline with {max_concurrent} max concurrent tasks", file=sys.stderr)
    sys.stderr.flush()
    
    try:
        async for task in loader:
            constructed += 1
            
            # Print EVERY construction for debugging
            print(
                f"[CONSTRUCTION] {constructed}/{loader.total_tasks} "
                f"({constructed*100//loader.total_tasks}%) | "
                f"Latent {task.record.latent.latent_index}",
                file=sys.stderr
            )
            sys.stderr.flush()
            
            # Create scoring task
            coro = asyncio.create_task(process_and_update(task))
            tasks.add(coro)
            
            # Limit concurrency
            if len(tasks) >= max_concurrent:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                results.extend(task.result() for task in done)
                tasks = pending
        
        # Wait for remaining tasks
        if tasks:
            print(f"\n[PIPELINE] Waiting for {len(tasks)} remaining scoring tasks...", file=sys.stderr)
            sys.stderr.flush()
            done, _ = await asyncio.wait(tasks)
            results.extend(task.result() for task in done)
    
    finally:
        total_time = time.time() - start_time
        avg_rate = scored / total_time if total_time > 0 else 0
        print(f"\n[COMPLETE] Scored {scored} labels in {total_time:.1f}s (avg {avg_rate:.1f}/s)", file=sys.stderr)
        sys.stderr.flush()
    
    return results


def compute_summary_statistics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute summary statistics across all scored labels.
    """
    # Overall statistics
    all_metrics = [r["metrics"] for r in results]
    
    summary = {
        "total_labels": len(results),
        "unique_latents": len(set(r["latent_index"] for r in results)),
        "overall": {
            "mean_f1": np.mean([m["f1"] for m in all_metrics]),
            "std_f1": np.std([m["f1"] for m in all_metrics]),
            "mean_accuracy": np.mean([m["accuracy"] for m in all_metrics]),
            "std_accuracy": np.std([m["accuracy"] for m in all_metrics]),
            "mean_precision": np.mean([m["precision"] for m in all_metrics]),
            "std_precision": np.std([m["precision"] for m in all_metrics]),
            "mean_recall": np.mean([m["recall"] for m in all_metrics]),
            "std_recall": np.std([m["recall"] for m in all_metrics]),
            "total_tp": sum(m["tp"] for m in all_metrics),
            "total_fp": sum(m["fp"] for m in all_metrics),
            "total_tn": sum(m["tn"] for m in all_metrics),
            "total_fn": sum(m["fn"] for m in all_metrics),
        }
    }
    
    # Per-scale statistics (if applicable)
    scales = set(r["scale"] for r in results if r["scale"] is not None)
    if scales:
        summary["by_scale"] = {}
        for scale in sorted(scales):
            scale_results = [r for r in results if r["scale"] == scale]
            scale_metrics = [r["metrics"] for r in scale_results]
            summary["by_scale"][str(scale)] = {
                "n_labels": len(scale_results),
                "mean_f1": np.mean([m["f1"] for m in scale_metrics]),
                "std_f1": np.std([m["f1"] for m in scale_metrics]),
                "mean_accuracy": np.mean([m["accuracy"] for m in scale_metrics]),
                "std_accuracy": np.std([m["accuracy"] for m in scale_metrics]),
                "mean_precision": np.mean([m["precision"] for m in scale_metrics]),
                "mean_recall": np.mean([m["recall"] for m in scale_metrics]),
            }
    
    return summary


async def main():
    parser = argparse.ArgumentParser(
        description="Score existing auto-interp labels with delphi scorers"
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
        help="Path to activations directory (default: neuronpedia_activations)"
    )
    parser.add_argument(
        "--module",
        type=str,
        default="19-llamascope-res-32k",
        help="Module name (default: 19-llamascope-res-32k)"
    )
    parser.add_argument(
        "--scorer",
        type=str,
        default="detection",
        choices=["detection"],
        help="Scorer to use (default: detection)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Max concurrent scoring tasks (default: 10)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model to use for scoring (default: meta-llama/Meta-Llama-3.1-8B-Instruct)"
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
        help="Max model length for vLLM (default: 4096)"
    )
    parser.add_argument(
        "--max-latents",
        type=int,
        default=None,
        help="Max number of unique latents to test (tests all labels/scales for selected latents)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Label Scoring Pipeline")
    print("="*80)
    print(f"Labels: {args.labels}")
    print(f"Activations: {args.activations}")
    print(f"Module: {args.module}")
    print(f"Scorer: {args.scorer}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Load labels
    labels = load_labels(args.labels)
    
    # Filter to max_latents if specified
    if args.max_latents is not None:
        # Get all unique latent indices (sorted for consistency)
        all_latent_indices = sorted(set(l.latent_index for l in labels))
        
        # Select first N latents
        selected_latents = set(all_latent_indices[:args.max_latents])
        
        # Filter labels to only include selected latents
        original_count = len(labels)
        labels = [l for l in labels if l.latent_index in selected_latents]
        
        print(f"\n⚠️  Limited to {args.max_latents} latents (out of {len(all_latent_indices)} total)")
        print(f"   Filtered from {original_count} to {len(labels)} labels")
        
        # Show scale breakdown if applicable
        if labels and labels[0].scale is not None:
            scales_per_latent = len(labels) / len(selected_latents)
            print(f"   Testing ~{scales_per_latent:.1f} labels per latent (different scales)")
    
    # Get unique latent indices
    latent_indices = list(set(l.latent_index for l in labels))
    
    # ⚠️ CRITICAL: Initialize vLLM BEFORE loading dataset
    # The LatentDataset initializes CUDA/PyTorch, which must happen AFTER vLLM
    # See VLLM_CUDA_FIX.md for details
    print(f"\n{'='*80}")
    print(f"Initializing vLLM (must happen before dataset loading)")
    print(f"{'='*80}")
    print(f"Loading model: {args.model}")
    print(f"Using {args.num_gpus} GPU(s)")
    
    client = Offline(
        args.model,
        max_memory=0.9,
        max_model_len=args.max_model_len,
        num_gpus=args.num_gpus,
    )
    
    print("✓ vLLM client initialized")
    
    scorer = DetectionScorer(
        client,
        n_examples_shown=5,  # Number of examples per prompt batch (more efficient than 1)
        verbose=False,
        log_prob=True,  # Get probabilities for more detailed analysis
    )
    
    print("✓ Scorer initialized")
    
    # Now safe to create dataset (which initializes CUDA/PyTorch)
    dataset = create_latent_dataset(
        args.activations,
        args.module,
        latent_indices
    )
    
    # Create loader
    loader = LabelScoringLoader(dataset, labels)
    
    # Run pipeline
    print(f"\n{'='*80}")
    print(f"Running Scoring Pipeline")
    print(f"{'='*80}")
    print(f"Max concurrent: {args.max_concurrent}")
    print()
    
    results = await run_scoring_pipeline(loader, scorer, args.max_concurrent)
    
    # Compute summary statistics
    print(f"\n{'='*80}")
    print(f"Computing Summary Statistics")
    print(f"{'='*80}")
    
    summary = compute_summary_statistics(results)
    
    # Save results
    results_file = args.output / "detailed_results.json"
    summary_file = args.output / "summary_statistics.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved detailed results to {results_file}")
    print(f"✓ Saved summary statistics to {summary_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary Statistics")
    print(f"{'='*80}")
    print(f"Total labels scored: {summary['total_labels']}")
    print(f"Unique latents: {summary['unique_latents']}")
    print(f"\nOverall Performance:")
    print(f"  Mean F1:        {summary['overall']['mean_f1']:.4f} ± {summary['overall']['std_f1']:.4f}")
    print(f"  Mean Accuracy:  {summary['overall']['mean_accuracy']:.4f} ± {summary['overall']['std_accuracy']:.4f}")
    print(f"  Mean Precision: {summary['overall']['mean_precision']:.4f} ± {summary['overall']['std_precision']:.4f}")
    print(f"  Mean Recall:    {summary['overall']['mean_recall']:.4f} ± {summary['overall']['std_recall']:.4f}")
    
    if "by_scale" in summary:
        print(f"\nPer-Scale Performance:")
        for scale, stats in sorted(summary["by_scale"].items(), key=lambda x: float(x[0])):
            print(f"  Scale {scale}:")
            print(f"    N labels:      {stats['n_labels']}")
            print(f"    Mean F1:       {stats['mean_f1']:.4f} ± {stats['std_f1']:.4f}")
            print(f"    Mean Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
    
    print(f"\n{'='*80}")
    print(f"✓ SCORING COMPLETE!")
    print(f"{'='*80}")
    
    # Clean up
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())

