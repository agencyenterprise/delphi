# Label Scoring Script

This document describes how to use `score_generated_labels.py` to efficiently score generated labels using the detection scorer with vLLM.

## Overview

The script scores all labels in a generated_labels JSON file by:
1. Initializing vLLM before loading PyTorch datasets (critical for avoiding CUDA deadlock)
2. Loading activation data from neuronpedia or delphi format
3. Batching labels across multiple latents for efficient scoring
4. Computing summary statistics overall and by scale parameter

## Prerequisites

Activate the virtual environment before running the script:

```bash
source /tmp/venv/bin/activate
```

## Usage

### Basic Usage

```bash
python score_generated_labels.py \
  --labels-file generated_labels_Llama-3.1-8B-res-32k_layer19_zero_bias.json \
  --activations-dir neuronpedia_activations/19-llamascope-res-32k \
  --output-file scored_labels_output.json
```

### Full Options

```bash
python score_generated_labels.py \
  --labels-file generated_labels_Llama-3.1-8B-res-32k_layer19_zero_bias.json \
  --activations-dir neuronpedia_activations/19-llamascope-res-32k \
  --output-file scored_labels_output.json \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --tokenizer meta-llama/Llama-3.1-8B \
  --num-gpus 2 \
  --max-model-len 4096 \
  --num-examples-per-prompt 5 \
  --batch-size 10
```

### Testing with Limited Latents

For testing, you can limit the number of latents processed:

```bash
python score_generated_labels.py \
  --labels-file generated_labels_Llama-3.1-8B-res-32k_layer19_zero_bias.json \
  --activations-dir neuronpedia_activations/19-llamascope-res-32k \
  --output-file scored_labels_test.json \
  --max-latents 10
```

## Arguments

- `--labels-file`: Path to the generated_labels JSON file (required)
- `--activations-dir`: Path to activations directory, e.g., `neuronpedia_activations/19-llamascope-res-32k` (required)
- `--output-file`: Path to output JSON file (required)
- `--model`: Model to use for scoring (default: `meta-llama/Meta-Llama-3.1-8B-Instruct`)
- `--tokenizer`: Tokenizer to use (defaults to same as `--model`)
- `--num-gpus`: Number of GPUs to use (default: 2)
- `--max-model-len`: Maximum model context length (default: 4096)
- `--num-examples-per-prompt`: Number of examples per scorer prompt (default: 5)
- `--batch-size`: Number of latents to process in parallel (default: 10)
- `--max-latents`: Maximum number of latents to score, for testing (optional)

## Output Format

The script produces a JSON file with the following structure:

```json
{
  "metadata": {
    "dataset_name": "LlamaScope/Llama-3.1-8B-res-32k",
    "layer": 19,
    "scale_values": [2.0, 3.0, 5.0, 8.0, 13.0],
    ...
  },
  "scores": [
    {
      "latent_index": 21,
      "label": "...",
      "scale": 2.0,
      "accuracy": 0.85,
      "num_correct": 17,
      "num_total": 20,
      "avg_probability": 0.78,
      "per_example_results": [...]
    },
    ...
  ],
  "summary_statistics": {
    "overall": {
      "avg_accuracy": 0.75,
      "avg_probability": 0.72,
      "total_labels": 5000,
      "min_accuracy": 0.20,
      "max_accuracy": 1.00
    },
    "by_scale": {
      "2.0": {
        "avg_accuracy": 0.70,
        "avg_probability": 0.68,
        "count": 1000,
        "min_accuracy": 0.20,
        "max_accuracy": 0.98
      },
      ...
    }
  }
}
```

## Performance Optimization

The script includes several optimizations:

1. **vLLM initialization before dataset loading** - Critical for avoiding CUDA deadlock (see `VLLM_CUDA_FIX.md`)
2. **Async/parallel scoring** - Batches multiple latents together
3. **Pre-loading LatentRecords** - Loads all needed records into memory once
4. **Efficient batching** - Leverages vLLM's internal batching for multiple prompts

## Important Notes

### vLLM + PyTorch CUDA Issue

**Always ensure vLLM is initialized before loading any PyTorch datasets!** This script follows the correct order:

1. ✅ Initialize vLLM client
2. ✅ Create DetectionScorer
3. ✅ Load tokenizer and LatentDataset

See `VLLM_CUDA_FIX.md` for more details on this critical requirement.

### Memory Usage

The script loads all needed LatentRecords into memory. For large numbers of latents, this may require significant RAM. Consider using `--max-latents` to process in chunks if needed.

### Activation Data Format

The script works with activation data in the neuronpedia format created by `download_neuronpedia_activations.py`. The expected directory structure:

```
neuronpedia_activations/
  19-llamascope-res-32k/
    0_6552.safetensors
    6553_13105.safetensors
    ...
    config.json
```

## Example Workflow

1. Download activations:
```bash
python download_neuronpedia_activations.py \
  --model llama3.1-8b \
  --sae 19-llamascope-res-32k \
  --num-workers 8
```

2. Score your labels:
```bash
python score_generated_labels.py \
  --labels-file generated_labels_Llama-3.1-8B-res-32k_layer19_zero_bias.json \
  --activations-dir neuronpedia_activations/19-llamascope-res-32k \
  --output-file scored_labels_output.json \
  --num-gpus 2
```

3. Analyze results in the output JSON file

