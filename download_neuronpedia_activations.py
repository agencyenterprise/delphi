#!/usr/bin/env python3
"""
Download and convert NeuronPedia activation data from S3 to Delphi format.

Uses parallel processing for fast downloads and parsing.

The script creates unique directories for each SAE:
    Output:
        19-llamascope-res-32k   → ./neuronpedia_activations/model.layers.19.res-32k/
        20-llamascope-res-131k  → ./neuronpedia_activations/model.layers.20.res-131k/
        27-llamascope-mlp-32k   → ./neuronpedia_activations/model.layers.27.mlp-32k/
    
    Cache:
        19-llamascope-res-32k   → ./neuronpedia_cache/19-llamascope-res-32k/
        20-llamascope-res-131k  → ./neuronpedia_cache/20-llamascope-res-131k/
        27-llamascope-mlp-32k   → ./neuronpedia_cache/27-llamascope-mlp-32k/

Usage:
    # Layer 19 residual SAE (32k features)
    python download_neuronpedia_activations.py \
        --model llama3.1-8b \
        --layer 19-llamascope-res-32k \
        --num-workers 8
    
    # Layer 20 residual SAE (131k features)
    python download_neuronpedia_activations.py \
        --model llama3.1-8b \
        --layer 20-llamascope-res-131k \
        --num-workers 8
    
    # Layer 27 MLP SAE (32k features)
    python download_neuronpedia_activations.py \
        --model llama3.1-8b \
        --layer 27-llamascope-mlp-32k \
        --num-workers 8
"""

import argparse
import gzip
import json
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoTokenizer


def list_s3_batches(bucket: str, prefix: str) -> List[str]:
    """
    List all batch files in the S3 bucket.
    
    For NeuronPedia, the URL pattern is:
    https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/v1/{model}/{layer}/activations/batch-{i}.jsonl.gz
    """
    base_url = f"https://{bucket}.s3.us-east-1.amazonaws.com/{prefix}"
    
    # NeuronPedia typically has batches numbered 0-31 (32 batches total)
    batch_urls = []
    for i in range(32):
        url = f"{base_url}/batch-{i}.jsonl.gz"
        batch_urls.append(url)
    
    return batch_urls


def download_file(url: str, output_path: Path) -> tuple[bool, str]:
    """Download a file from URL to output_path."""
    try:
        urllib.request.urlretrieve(url, output_path)
        return True, str(output_path)
    except Exception as e:
        return False, f"Failed to download {url}: {e}"


def parse_neuronpedia_batch(
    batch_file: Path,
    tokenizer,
    max_ctx_len: int = 128
) -> Dict[int, Dict[str, List]]:
    """
    Parse a NeuronPedia batch file and extract activation data.
    
    Returns:
        Dictionary mapping feature_idx -> {activations, locations, token_sequences}
    """
    feature_data = defaultdict(lambda: {
        'activations': [],
        'locations': [],
        'token_sequences': []  # List of token sequences
    })
    
    with gzip.open(batch_file, 'rt') as f:
        for line in f:
            record = json.loads(line)
            feature_idx = int(record['index'])
            
            # Get tokens and values
            tokens = record['tokens']
            values = record['values']
            
            # Tokenize the text
            # Concatenate tokens to get the original text
            text = ''.join(tokens)
            
            # Use the tokenizer to get token IDs
            token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_ctx_len)
            
            # For now, we'll store the sequence
            seq_idx = len(feature_data[feature_idx]['token_sequences'])
            feature_data[feature_idx]['token_sequences'].append(token_ids)
            
            # Find positions where the feature activates
            # Match NeuronPedia tokens to tokenizer tokens (approximately)
            # This is tricky because NeuronPedia uses different tokenization
            # For simplicity, we'll map proportionally
            
            if len(token_ids) > 0:
                ratio = len(token_ids) / len(values)
                
                for pos, val in enumerate(values):
                    if val > 0:  # Activation threshold
                        # Map NeuronPedia position to our tokenizer position
                        mapped_pos = min(int(pos * ratio), len(token_ids) - 1)
                        
                        feature_data[feature_idx]['activations'].append(float(val))
                        # locations format: [seq_idx, position, feature_idx]
                        feature_data[feature_idx]['locations'].append([
                            seq_idx,
                            mapped_pos,
                            feature_idx
                        ])
    
    return dict(feature_data)


def parse_batch_worker(args):
    """Worker function for parallel batch parsing."""
    batch_file, tokenizer_name, max_ctx_len = args
    # Load tokenizer in worker (can't pickle tokenizer objects)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return batch_file.name, parse_neuronpedia_batch(batch_file, tokenizer, max_ctx_len)


def merge_feature_data(
    all_data: Dict[int, Dict[str, List]],
    new_data: Dict[int, Dict[str, List]]
) -> Dict[int, Dict[str, List]]:
    """Merge new feature data into existing data."""
    for feature_idx, data in new_data.items():
        if feature_idx not in all_data:
            all_data[feature_idx] = {
                'activations': [],
                'locations': [],
                'token_sequences': []
            }
        
        # Offset sequence indices in locations
        seq_offset = len(all_data[feature_idx]['token_sequences'])
        
        all_data[feature_idx]['activations'].extend(data['activations'])
        all_data[feature_idx]['token_sequences'].extend(data['token_sequences'])
        
        # Adjust sequence indices in locations
        for loc in data['locations']:
            all_data[feature_idx]['locations'].append([
                loc[0] + seq_offset,
                loc[1],
                loc[2]
            ])
    
    return all_data


def save_as_safetensors(
    feature_data: Dict[int, Dict[str, List]],
    output_dir: Path,
    shard_size: int = 6553
):
    """
    Save feature data as safetensors shards.
    
    Args:
        feature_data: Dictionary of feature activations
        output_dir: Output directory
        shard_size: Number of features per shard
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine the range of features
    if not feature_data:
        print("No feature data to save!")
        return
    
    min_feat = min(feature_data.keys())
    max_feat = max(feature_data.keys())
    
    print(f"Saving features {min_feat} to {max_feat} in shards of {shard_size}...")
    
    # Create shards
    for shard_start in range(0, max_feat + 1, shard_size):
        shard_end = min(shard_start + shard_size - 1, max_feat)
        
        # Collect all tokens for this shard
        all_tokens = []
        all_activations = []
        all_locations = []
        
        token_offset = 0
        
        for feat_idx in range(shard_start, shard_end + 1):
            if feat_idx in feature_data:
                data = feature_data[feat_idx]
                
                # Add tokens
                for token_seq in data['token_sequences']:
                    all_tokens.append(token_seq)
                
                # Add activations
                all_activations.extend(data['activations'])
                
                # Add locations with adjusted sequence indices
                for loc in data['locations']:
                    all_locations.append([
                        loc[0] + token_offset,  # Adjust sequence index
                        loc[1],  # Position within sequence
                        loc[2] - shard_start  # Feature index (relative to shard start)
                    ])
                
                token_offset += len(data['token_sequences'])
        
        if not all_activations:
            print(f"Skipping empty shard {shard_start}_{shard_end}")
            continue
        
        # Pad all token sequences to the same length
        if all_tokens:
            max_len = max(len(seq) for seq in all_tokens)
            padded_tokens = []
            for seq in all_tokens:
                padded = seq + [0] * (max_len - len(seq))
                padded_tokens.append(padded)
            
            tokens_tensor = torch.tensor(padded_tokens, dtype=torch.int64)
        else:
            tokens_tensor = torch.zeros((0, 128), dtype=torch.int64)
        
        # Create tensors
        shard_data = {
            'activations': torch.tensor(all_activations, dtype=torch.float32),
            'locations': torch.tensor(all_locations, dtype=torch.int64),
            'tokens': tokens_tensor
        }
        
        # Save
        output_file = output_dir / f"{shard_start}_{shard_end}.safetensors"
        save_file(shard_data, output_file)
        print(f"Saved {output_file} with {len(all_activations)} activations")


def save_config(output_dir: Path, model_name: str, ctx_len: int):
    """Save config.json for Delphi."""
    config = {
        "model_name": model_name,
        "ctx_len": ctx_len,
        "dataset_repo": "neuronpedia",
        "dataset_split": "activations",
        "dataset_name": None,
        "dataset_column": "raw_content"
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved config to {output_dir / 'config.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert NeuronPedia activations to Delphi format"
    )
    parser.add_argument(
        "--model",
        default="llama3.1-8b",
        help="Model name in NeuronPedia (e.g., llama3.1-8b)"
    )
    parser.add_argument(
        "--layer",
        default="19-llamascope-res-32k",
        help="Layer/SAE identifier (e.g., 19-llamascope-res-32k, 20-llamascope-res-131k, 27-llamascope-mlp-32k)"
    )
    parser.add_argument(
        "--output-dir",
        default="./neuronpedia_activations",
        help="Output directory for converted data"
    )
    parser.add_argument(
        "--tokenizer",
        default="meta-llama/Llama-3.1-8B",
        help="HuggingFace tokenizer to use"
    )
    parser.add_argument(
        "--cache-dir",
        default="./neuronpedia_cache",
        help="Directory to cache downloaded files"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches to process (for testing)"
    )
    parser.add_argument(
        "--ctx-len",
        type=int,
        default=128,
        help="Context length for tokenization"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for downloading and parsing (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Parse layer string to create a unique module name
    # E.g., "20-llamascope-res-131k" -> "model.layers.20.res-131k"
    # E.g., "27-llamascope-mlp-32k" -> "model.layers.27.mlp-32k"
    parts = args.layer.split('-')
    layer_num = parts[0]
    
    # Extract SAE type (everything after "llamascope-")
    # "20-llamascope-res-131k" -> ["20", "llamascope", "res", "131k"]
    if len(parts) >= 3 and parts[1] == "llamascope":
        sae_type = '-'.join(parts[2:])  # "res-131k", "mlp-32k", etc.
        module_name = f"model.layers.{layer_num}.{sae_type}"
    else:
        # Fallback for non-standard naming
        module_name = f"model.layers.{layer_num}"
    
    # Setup paths - make cache SAE-specific too
    output_dir = Path(args.output_dir) / module_name
    cache_dir = Path(args.cache_dir) / args.layer  # Use full layer identifier for cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Model: {args.model}")
    print(f"Layer/SAE: {args.layer}")
    print(f"Module name: {module_name}")
    print(f"Cache directory: {cache_dir}")
    print(f"Output directory: {output_dir}")
    
    # List batches
    bucket = "neuronpedia-datasets"
    prefix = f"v1/{args.model}/{args.layer}/activations"
    
    batch_urls = list_s3_batches(bucket, prefix)
    
    if args.max_batches:
        batch_urls = batch_urls[:args.max_batches]
    
    print(f"Found {len(batch_urls)} batches to process")
    print(f"Using {args.num_workers} parallel workers")
    
    # Step 1: Download batches in parallel (I/O bound - use threads)
    print("\n" + "=" * 80)
    print("STEP 1: Downloading batches")
    print("=" * 80)
    
    download_tasks = []
    for url in batch_urls:
        batch_name = url.split('/')[-1]
        cache_file = cache_dir / batch_name
        
        # Skip if already downloaded
        if not cache_file.exists():
            download_tasks.append((url, cache_file))
    
    if download_tasks:
        print(f"\nDownloading {len(download_tasks)} batches...")
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(download_file, url, cache_file): (url, cache_file)
                for url, cache_file in download_tasks
            }
            
            with tqdm(total=len(futures), desc="Downloading") as pbar:
                for future in as_completed(futures):
                    success, msg = future.result()
                    if not success:
                        print(f"\n{msg}")
                    pbar.update(1)
    else:
        print("All batches already cached, skipping download")
    
    # Step 2: Parse batches in parallel (CPU bound - use processes)
    print("\n" + "=" * 80)
    print("STEP 2: Parsing batches")
    print("=" * 80)
    
    # Collect all cached batch files
    batch_files = []
    for url in batch_urls:
        batch_name = url.split('/')[-1]
        cache_file = cache_dir / batch_name
        if cache_file.exists():
            batch_files.append(cache_file)
    
    print(f"\nParsing {len(batch_files)} batches in parallel...")
    
    all_feature_data = {}
    
    # Prepare arguments for workers
    parse_args = [(bf, args.tokenizer, args.ctx_len) for bf in batch_files]
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(parse_batch_worker, arg): arg[0]
            for arg in parse_args
        }
        
        with tqdm(total=len(futures), desc="Parsing") as pbar:
            for future in as_completed(futures):
                batch_name, batch_data = future.result()
                
                # Merge into all data
                all_feature_data = merge_feature_data(all_feature_data, batch_data)
                
                pbar.set_postfix({"features": len(all_feature_data)})
                pbar.update(1)
    
    print(f"\n✅ Processed {len(batch_files)} batches, total features: {len(all_feature_data)}")
    
    # Step 3: Save as safetensors
    print("\n" + "=" * 80)
    print("STEP 3: Saving to safetensors format")
    print("=" * 80)
    save_as_safetensors(all_feature_data, output_dir)
    
    # Save config
    save_config(output_dir, args.tokenizer, args.ctx_len)
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Total features with activations: {len(all_feature_data)}")
    print(f"Total batches processed: {len(batch_files)}")


if __name__ == "__main__":
    main()

