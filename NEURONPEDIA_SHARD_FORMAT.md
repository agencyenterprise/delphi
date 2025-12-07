# Delphi Shard Format for NeuronPedia Data

## The Core Requirement

When converting NeuronPedia activation data to delphi's safetensor format, **every shard must contain the SAME complete token dataset**. This is not optional for the latents pre-filtering feature to work.

## Why This Matters

Delphi's `LatentDataset` has a `latents=` parameter that enables pre-filtering: loading only specific latents instead of iterating through all 32k features. This is critical for performance when you only need a subset of features.

When pre-filtering is enabled, `_build_selected()` loads only the shard files that contain your requested latents. The first loaded shard provides the token dataset via `self.tokens = tensor_buffer.tokens`.

**The Problem**: If each shard contains different tokens, you get an index mismatch:
- Activation locations reference global sequence indices (e.g., sequence 1,406,132)
- But the loaded shard only has tokens for that shard's features (e.g., 351,253 sequences)
- Result: `IndexError: index out of bounds`

## The Correct Structure

### Token Dataset
Build ONE global token dataset from all features across all batches. This becomes a shared reference that every shard includes.

### Shard Contents
Each shard file must contain:
1. **tokens**: The complete global token dataset (identical across all shards)
2. **activations**: Only the activations for features in this shard's range
3. **locations**: Activation locations with:
   - Sequence index: GLOBAL (references the shared token dataset)
   - Position: Token position within the sequence
   - Feature index: RELATIVE to shard start (delphi adds `first_latent` automatically)

### Example
For a dataset with 2.1M total token sequences split into 6 shards:

**Shard 0_6552.safetensors:**
- tokens: [2.1M x max_len] ← Full dataset
- activations: [N activations from features 0-6552]
- locations: [[global_seq_idx, pos, feat_idx - 0], ...]

**Shard 6553_13105.safetensors:**
- tokens: [2.1M x max_len] ← Same full dataset
- activations: [M activations from features 6553-13105]  
- locations: [[global_seq_idx, pos, feat_idx - 6553], ...]

## Why Relative Feature Indices

Delphi's `TensorBuffer.load()` parses the shard filename to get `first_latent`, then adjusts feature indices:

```
locations[:, 2] = locations[:, 2] + first_latent
```

So if you store absolute feature indices, they get double-counted. Store them relative to the shard's start.

## Implementation Strategy

1. **First pass**: Collect all token sequences from all features/batches into a global list
2. **Create mapping**: Map (feature_idx, local_seq_idx) → global_seq_idx
3. **Convert to tensor**: Pad and tensorize the global token dataset once
4. **Per shard**: 
   - Include the same global token tensor
   - Collect activations for features in this shard's range
   - Convert location sequence indices from local to global using the mapping
   - Make feature indices relative to shard start

## Verification

After generating shards, verify the format:
- All shard files should have identical `tokens` tensor shapes
- Loading any single shard via `LatentDataset(latents=...)` should work without index errors
- The total dataset size ≈ (global_tokens_size + activations_per_shard_size) × num_shards

## Performance Impact

**Without pre-filtering** (wrong format or disabled):
- Loads all 32k latents, filters during iteration
- ~30-60 minutes to get 3k latents

**With pre-filtering** (correct format):
- Loads only the requested latents' shards
- ~10-30 seconds for 3k latents
- **100x speedup**

## Key Takeaway

Think of the token dataset as a **shared global reference** that all shards point into, not as shard-specific data. Each shard gets a complete copy of this reference so it can resolve any activation location independently.

