#!/usr/bin/env python3
"""
Convert NeuronPedia activation data to delphi's safetensor format.

This script downloads activation data from NeuronPedia's S3 bucket and converts
it to the format expected by delphi's LatentDataset, following the critical
requirement that every shard must contain the SAME complete token dataset.

Usage:
    python convert_neuronpedia_activations.py --model llama3.1-8b --layer 19-llamascope-res-32k
"""

import argparse
import gzip
import json
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from safetensors.numpy import save_file
from tqdm import tqdm
from transformers import AutoTokenizer


def download_s3_files(
    model_id: str,
    layer: str,
    cache_dir: Path,
    max_batches: Optional[int] = None,
    force_download: bool = False
) -> List[Path]:
    """
    Download activation files from NeuronPedia S3 bucket.
    
    Args:
        model_id: Model identifier (e.g., "llama3.1-8b")
        layer: Layer identifier (e.g., "19-llamascope-res-32k")
        cache_dir: Directory to cache downloaded files
        max_batches: Maximum number of batches to download (None = all)
        force_download: If True, re-download even if files exist locally
    
    Returns:
        List of paths to downloaded batch files
    """
    layer_cache_dir = cache_dir / layer
    layer_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # S3 URL pattern
    s3_base = f"https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/v1/{model_id}/{layer}/activations/"
    
    # First, check what's already downloaded
    existing_files = sorted(layer_cache_dir.glob("batch-*.jsonl.gz"))
    
    if existing_files and not force_download:
        if max_batches:
            existing_files = existing_files[:max_batches]
        print(f"✓ Found {len(existing_files)} cached batch files (use --force-download to re-download)")
        return existing_files
    
    if force_download and existing_files:
        print(f"Force download enabled, will re-download {len(existing_files)} existing files")
    
    # Download files using wget or curl
    print(f"Downloading activation data from NeuronPedia S3...")
    print(f"Model: {model_id}, Layer: {layer}")
    
    downloaded_files = []
    batch_idx = 0
    
    while True:
        if max_batches and batch_idx >= max_batches:
            break
            
        batch_file = f"batch-{batch_idx}.jsonl.gz"
        local_path = layer_cache_dir / batch_file
        
        # Skip if already exists
        if local_path.exists():
            print(f"✓ {batch_file} (cached)")
            downloaded_files.append(local_path)
            batch_idx += 1
            continue
        
        # Try to download
        url = s3_base + batch_file
        try:
            result = subprocess.run(
                ["wget", "-q", "--spider", url],
                capture_output=True,
                timeout=10
            )
            
            if result.returncode != 0:
                # File doesn't exist on S3
                if batch_idx == 0:
                    raise RuntimeError(f"No data found at {s3_base}")
                print(f"Downloaded {batch_idx} batch files total")
                break
            
            # Download the file
            print(f"⬇ Downloading {batch_file}...")
            subprocess.run(
                ["wget", "-q", "-O", str(local_path), url],
                check=True,
                timeout=300
            )
            downloaded_files.append(local_path)
            batch_idx += 1
            
        except subprocess.TimeoutExpired:
            print(f"⚠ Timeout downloading {batch_file}, stopping")
            break
        except subprocess.CalledProcessError as e:
            print(f"⚠ Error downloading {batch_file}: {e}")
            break
        except Exception as e:
            print(f"⚠ Unexpected error: {e}")
            break
    
    return downloaded_files


def process_batch_file(batch_file: Path) -> Tuple[List[List[str]], Dict[int, List], Dict[int, int]]:
    """
    Process a single batch file in parallel.
    
    Returns:
        Tuple of:
        - token_strings: List of token string sequences
        - batch_feature_data: Dict mapping feature_idx -> list of (batch_local_seq_idx, activations)
        - batch_feature_counts: Dict mapping feature_idx -> count for this batch
    """
    token_strings = []
    batch_feature_data = defaultdict(list)
    batch_feature_counts = defaultdict(int)
    
    with gzip.open(batch_file, 'rt') as f:
        for line in f:
            record = json.loads(line)
            feature_idx = int(record['index'])
            tokens = record['tokens']
            values = record['values']
            
            # Get sequence index within this batch
            batch_seq_idx = len(token_strings)
            token_strings.append(tokens)
            
            # Store activation data with batch-local sequence index
            activations = []
            for pos, val in enumerate(values):
                if val > 0:
                    activations.append((pos, val))
            
            batch_feature_data[feature_idx].append((batch_seq_idx, activations))
            batch_feature_counts[feature_idx] += 1
    
    return token_strings, batch_feature_data, batch_feature_counts


def load_neuronpedia_data(
    batch_files: List[Path],
    tokenizer_name: str = "meta-llama/Llama-3.1-8B",
    num_workers: Optional[int] = None
) -> Tuple[torch.Tensor, Dict[int, List], Dict[Tuple[int, int], int]]:
    """
    Load NeuronPedia activation data and build global token dataset.
    
    This is the critical first pass that collects ALL token sequences from
    ALL features to create the shared global reference.
    
    Args:
        batch_files: List of paths to batch-*.jsonl.gz files
        tokenizer_name: HuggingFace tokenizer name
        num_workers: Number of parallel workers (None = use CPU count)
    
    Returns:
        Tuple of:
        - global_tokens: Tensor of shape [num_sequences, max_seq_len]
        - feature_data: Dict mapping feature_idx -> list of (local_seq_idx, activations)
        - global_seq_mapping: Dict mapping (feature_idx, local_seq_idx) -> global_seq_idx
    """
    print("\n" + "="*80)
    print("PHASE 1: Building global token dataset")
    print("="*80)
    
    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, len(batch_files))
    else:
        num_workers = min(num_workers, len(batch_files))
    
    print(f"Using {num_workers} parallel workers")
    
    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Process batch files in parallel
    print("\nFirst pass: Collecting all token sequences (parallel)...")
    
    global_token_strings = []
    global_seq_mapping = {}
    feature_data = defaultdict(list)
    feature_local_counts = defaultdict(int)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all batch files for processing
        future_to_batch = {
            executor.submit(process_batch_file, batch_file): batch_file 
            for batch_file in batch_files
        }
        
        # Collect results as they complete
        with tqdm(total=len(batch_files), desc="Processing batches") as pbar:
            for future in as_completed(future_to_batch):
                batch_file = future_to_batch[future]
                try:
                    token_strings, batch_feature_data, batch_feature_counts = future.result()
                    
                    # Merge into global structures
                    batch_offset = len(global_token_strings)
                    
                    for feature_idx, sequences in batch_feature_data.items():
                        for batch_seq_idx, activations in sequences:
                            # Get local sequence index for this feature
                            local_seq_idx = feature_local_counts[feature_idx]
                            feature_local_counts[feature_idx] += 1
                            
                            # Map to global sequence index
                            global_seq_idx = batch_offset + batch_seq_idx
                            global_seq_mapping[(feature_idx, local_seq_idx)] = global_seq_idx
                            
                            # Store activation data
                            feature_data[feature_idx].append((local_seq_idx, activations))
                    
                    # Add token strings to global list
                    global_token_strings.extend(token_strings)
                    
                except Exception as e:
                    print(f"\n⚠ Error processing {batch_file}: {e}")
                
                pbar.update(1)
    
    total_sequences = len(global_token_strings)
    print(f"\n✓ Collected {total_sequences:,} token sequences")
    print(f"✓ Found {len(feature_data)} features")
    
    # Find max sequence length
    max_seq_len = max(len(tokens) for tokens in global_token_strings)
    print(f"✓ Max sequence length: {max_seq_len}")
    
    # Build token string -> token ID lookup table
    # NeuronPedia provides DECODED token strings, so we need to build a reverse mapping
    print("\nBuilding token lookup table...")
    print("  Creating decoded_string -> token_id mapping...")
    
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    
    # Build reverse mapping: decoded_string -> token_id
    decoded_to_id = {}
    for token_id in tqdm(range(vocab_size), desc="  Building lookup"):
        try:
            decoded = tokenizer.decode([token_id])
            decoded_to_id[decoded] = token_id
        except:
            pass  # Skip any problematic token IDs
    
    print(f"✓ Built lookup table with {len(decoded_to_id):,} decoded tokens")
    
    # Convert token strings to token IDs (using fast lookups)
    print("\nConverting tokens to IDs...")
    global_tokens = torch.zeros((total_sequences, max_seq_len), dtype=torch.int64)
    
    unknown_tokens = set()
    for seq_idx, token_strings in enumerate(tqdm(global_token_strings, desc="Tokenizing")):
        token_ids = []
        for token_str in token_strings:
            # Direct lookup (O(1) operation)
            if token_str in decoded_to_id:
                token_ids.append(decoded_to_id[token_str])
            else:
                # Fallback: encode the string (slower but handles edge cases)
                encoded = tokenizer.encode(token_str, add_special_tokens=False)
                if len(encoded) > 0:
                    token_ids.append(encoded[0])
                    # Cache it for future use
                    decoded_to_id[token_str] = encoded[0]
                else:
                    if token_str not in unknown_tokens:
                        unknown_tokens.add(token_str)
                    token_ids.append(tokenizer.unk_token_id or 0)
        
        seq_len = len(token_ids)
        global_tokens[seq_idx, :seq_len] = torch.tensor(token_ids, dtype=torch.int64)
    
    if unknown_tokens:
        print(f"⚠ Warning: {len(unknown_tokens)} unique token strings not in vocabulary (using UNK)")
        if len(unknown_tokens) <= 10:
            print(f"  Examples: {list(unknown_tokens)[:10]}")
    
    print(f"✓ Global token tensor shape: {global_tokens.shape}")
    
    return global_tokens, feature_data, global_seq_mapping


def create_single_shard(
    shard_idx: int,
    shard_size: int,
    max_feature_idx: int,
    global_tokens_np: np.ndarray,
    feature_data: Dict[int, List],
    global_seq_mapping: Dict[Tuple[int, int], int],
    layer_output_dir: Path
) -> Tuple[str, int, float]:
    """Create a single shard file (for parallel processing)."""
    start_feature = shard_idx * shard_size
    end_feature = min((shard_idx + 1) * shard_size - 1, max_feature_idx)
    
    # Collect activations and locations for this shard
    shard_activations = []
    shard_locations = []
    
    for feature_idx in range(start_feature, end_feature + 1):
        if feature_idx not in feature_data:
            continue
        
        relative_feature_idx = feature_idx - start_feature
        
        for local_seq_idx, activations in feature_data[feature_idx]:
            # Convert local sequence index to global
            global_seq_idx = global_seq_mapping[(feature_idx, local_seq_idx)]
            
            # Add each activation
            for pos, val in activations:
                shard_activations.append(val)
                shard_locations.append([global_seq_idx, pos, relative_feature_idx])
    
    if len(shard_activations) == 0:
        return f"{start_feature}_{end_feature}", 0, 0.0
    
    # Convert to numpy arrays
    activations_np = np.array(shard_activations, dtype=np.float32)
    locations_np = np.array(shard_locations, dtype=np.int32)
    
    # Save shard
    shard_path = layer_output_dir / f"{start_feature}_{end_feature}.safetensors"
    save_file(
        {
            "tokens": global_tokens_np,
            "activations": activations_np,
            "locations": locations_np,
        },
        str(shard_path),
    )
    
    shard_size_mb = shard_path.stat().st_size / (1024 * 1024)
    return f"{start_feature}_{end_feature}", len(shard_activations), shard_size_mb


def create_shards(
    global_tokens: torch.Tensor,
    feature_data: Dict[int, List],
    global_seq_mapping: Dict[Tuple[int, int], int],
    output_dir: Path,
    layer: str,
    shard_size: int = 6553,  # ~32768 / 5 shards
    num_workers: Optional[int] = None
):
    """
    Create safetensor shard files following delphi's format.
    
    Each shard contains:
    - tokens: The SAME complete global token dataset
    - activations: Only activations for features in this shard's range
    - locations: [global_seq_idx, pos, relative_feature_idx]
    
    Args:
        global_tokens: Tensor of shape [num_sequences, max_seq_len]
        feature_data: Dict mapping feature_idx -> list of (local_seq_idx, activations)
        global_seq_mapping: Dict mapping (feature_idx, local_seq_idx) -> global_seq_idx
        output_dir: Directory to write shard files
        layer: Layer identifier for subdirectory
        shard_size: Number of features per shard
        num_workers: Number of parallel workers
    """
    print("\n" + "="*80)
    print("PHASE 2: Creating safetensor shards")
    print("="*80)
    
    layer_output_dir = output_dir / layer
    layer_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine shard boundaries
    max_feature_idx = max(feature_data.keys())
    print(f"\nFeature range: 0 to {max_feature_idx}")
    print(f"Shard size: {shard_size} features")
    
    # Create shards
    num_shards = (max_feature_idx + shard_size) // shard_size
    print(f"Creating {num_shards} shards...")
    
    global_tokens_np = global_tokens.numpy().astype(np.int32)
    
    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, num_shards)
    else:
        num_workers = min(num_workers, num_shards)
    
    print(f"Using {num_workers} parallel workers for shard creation")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for shard_idx in range(num_shards):
            future = executor.submit(
                create_single_shard,
                shard_idx,
                shard_size,
                max_feature_idx,
                global_tokens_np,
                feature_data,
                global_seq_mapping,
                layer_output_dir
            )
            futures.append(future)
        
        # Collect results
        with tqdm(total=num_shards, desc="Writing shards") as pbar:
            for future in as_completed(futures):
                try:
                    shard_name, num_activations, size_mb = future.result()
                    if num_activations > 0:
                        print(f"  ✓ {shard_name}.safetensors "
                              f"({num_activations:,} activations, {size_mb:.1f} MB)")
                    else:
                        print(f"  ⚠ {shard_name} has no activations, skipped")
                except Exception as e:
                    print(f"  ✗ Error creating shard: {e}")
                pbar.update(1)
    
    # Create config.json
    config = {
        "ctx_len": global_tokens.shape[1],
        "dataset_repo": "neuronpedia",
        "dataset_split": "activations",
        "dataset_name": layer,
        "dataset_column": "text",
    }
    
    config_path = layer_output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Created config.json")
    print(f"✓ All shards written to {layer_output_dir}")
    
    # Verification
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    verify_shards(layer_output_dir, global_tokens.shape)


def verify_shards(output_dir: Path, expected_tokens_shape: tuple):
    """
    Verify that all shards have identical token tensor shapes.
    
    Args:
        output_dir: Directory containing shard files
        expected_tokens_shape: Expected shape of tokens tensor
    """
    shard_files = sorted(output_dir.glob("*.safetensors"))
    
    print(f"Checking {len(shard_files)} shard files...")
    
    from safetensors.numpy import load_file
    
    for shard_file in shard_files:
        data = load_file(str(shard_file))
        
        if "tokens" not in data:
            print(f"✗ {shard_file.name}: Missing 'tokens' key")
            continue
        
        tokens_shape = data["tokens"].shape
        if tokens_shape != expected_tokens_shape:
            print(f"✗ {shard_file.name}: Token shape mismatch")
            print(f"  Expected: {expected_tokens_shape}")
            print(f"  Got: {tokens_shape}")
        else:
            print(f"✓ {shard_file.name}: Token shape correct {tokens_shape}")
    
    print("\n✓ Verification complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NeuronPedia activations to delphi format"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1-8b",
        help="Model identifier (default: llama3.1-8b)"
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="19-llamascope-res-32k",
        help="Layer identifier (default: 19-llamascope-res-32k)"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("neuronpedia_cache"),
        help="Directory for cached downloads (default: neuronpedia_cache)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("neuronpedia_activations"),
        help="Directory for processed activations (default: neuronpedia_activations)"
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=6553,
        help="Number of features per shard (default: 6553)"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches to process (default: all)"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if files exist in cache (default: use cache)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("NeuronPedia to Delphi Activation Converter")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Shard size: {args.shard_size}")
    
    # Download files (or use cached)
    print()
    batch_files = download_s3_files(
        args.model,
        args.layer,
        args.cache_dir,
        args.max_batches,
        args.force_download
    )
    
    if not batch_files:
        raise RuntimeError("No batch files available")
    
    # Load data and build global token dataset
    global_tokens, feature_data, global_seq_mapping = load_neuronpedia_data(
        batch_files,
        num_workers=args.num_workers
    )
    
    # Create shards
    create_shards(
        global_tokens,
        feature_data,
        global_seq_mapping,
        args.output_dir,
        args.layer,
        args.shard_size,
        args.num_workers
    )
    
    print("\n" + "="*80)
    print("✓ CONVERSION COMPLETE!")
    print("="*80)
    print(f"Processed activations are in: {args.output_dir / args.layer}")
    print(f"You can now use this with LatentDataset by passing:")
    print(f'  raw_dir="{args.output_dir}"')
    print(f'  modules=["{args.layer}"]')


if __name__ == "__main__":
    main()

