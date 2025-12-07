# vLLM + PyTorch CUDA Initialization Issue

## The Problem

When PyTorch initializes CUDA (e.g., by loading tensors or models), it creates a CUDA context in the main process. If vLLM then tries to initialize, it spawns worker processes that attempt to create their own CUDA contexts, causing a **deadlock/hang** due to CUDA's incompatibility with fork().

## The Solution

**Always initialize vLLM BEFORE any PyTorch operations that touch the GPU.**

### Bad Order (Hangs):
```python
# Load dataset first (initializes CUDA)
dataset = LatentDataset(...)  # ❌ CUDA initialized here

# Then try to initialize vLLM
client = Offline(...)  # ⏸️ Hangs during worker spawn
```

### Good Order (Works):
```python
# Initialize vLLM first
client = Offline(...)  # ✅ Gets clean CUDA environment

# Then load dataset
dataset = LatentDataset(...)  # ✅ Works fine
```

## Key Rule

If you're using both vLLM and PyTorch in the same script, **vLLM must initialize first** before any CUDA operations (loading models, creating tensors on GPU, etc.).
