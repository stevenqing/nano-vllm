# Installation Guide

This guide covers installation of nano-vllm for various environments.

## Requirements

- Python 3.10 - 3.12
- PyTorch >= 2.4.0
- CUDA compatible GPU
- Triton >= 3.0.0
- Flash Attention

## Quick Installation (Standard Setup)

For most users with x86_64 systems and standard CUDA setup:

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Dependencies

nano-vllm requires the following packages (installed automatically):

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >= 2.4.0 | Deep learning framework |
| triton | >= 3.0.0 | Kernel compilation |
| transformers | >= 4.51.0 | Model loading |
| flash-attn | latest | Flash Attention kernels |
| xxhash | latest | Fast hashing for prefix caching |

## Development Installation

Clone and install in development mode:

```bash
git clone https://github.com/GeeeekExplorer/nano-vllm.git
cd nano-vllm
pip install -e .
```

## Model Download

Download model weights using Hugging Face CLI:

```bash
# Example: Qwen3-0.6B
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## HPC/SLURM Installation (ARM64/GH200)

For HPC clusters with ARM64 architecture and NVIDIA GH200 GPUs, follow this multi-step process.

### Step 1: Create Conda Environment

```bash
conda create -n nano python=3.11
conda activate nano
```

### Step 2: Environment Setup

Source the environment setup script before any installation:

```bash
source scripts/setup_env.sh nano
```

This script:
- Loads required modules (gcc-native/12.3, cuda/12.6)
- Sets CUDA paths and compiler environment
- Configures CUDA architecture for GH200 (SM90)
- Sets up torch extensions cache

### Step 3: Install Base Dependencies

```bash
pip install -U pip setuptools wheel packaging ninja cmake
pip install torch torchvision torchaudio
pip install transformers xxhash
```

### Step 4: Install Triton (ARM64)

Triton may need to be built from source on ARM64 systems:

```bash
# Submit SLURM job for Triton build
sbatch scripts/install_triton_nano.slurm
```

Or build manually:

```bash
# Try PyPI first
pip install triton==3.1.0

# If PyPI fails, build from source
git clone --recursive -b v3.1.0 https://github.com/triton-lang/triton.git
cd triton/python
export MAX_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1
pip install -v --no-build-isolation --no-cache-dir .
```

### Step 5: Install Flash Attention (ARM64)

Flash Attention requires compilation from source on ARM64:

```bash
# Submit SLURM job (takes 18+ hours!)
sbatch scripts/install_flash_attn.slurm
```

Key environment variables for ARM64 build:

```bash
export TORCH_CUDA_ARCH_LIST="9.0;9.0a"
export MAX_JOBS=1  # Critical for ARM64 - parallel builds fail
export CMAKE_BUILD_PARALLEL_LEVEL=1
pip install -v --no-build-isolation --no-cache-dir flash-attn
```

### Step 6: Install nano-vllm

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
# Or for development
pip install -e .
```

### Verification

Verify your installation:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import triton; print(f'Triton: {triton.__version__}')"
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"
python -c "from nanovllm import LLM; print('nano-vllm: OK')"
```

## vLLM Installation (for benchmarking)

To compare performance with vLLM, install it in a separate conda environment:

```bash
# Create separate environment
conda create -n vllm python=3.11
conda activate vllm

# For ARM64/GH200, use the install script
bash scripts/install_vllm.sh

# Or submit as SLURM job
sbatch scripts/install_vllm.slurm
```

## Quick Start

After installation, verify with a simple test:

```python
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Initialize
path = "~/huggingface/Qwen3-0.6B/"
tokenizer = AutoTokenizer.from_pretrained(path)
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

# Generate
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM!"]
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for p in prompts
]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])
```

## Troubleshooting

### GLIBCXX version errors

Add the correct libstdc++ to your library path:

```bash
export CXX=$(which g++)
LIBSTDCXX_SO="$($CXX -print-file-name=libstdc++.so.6)"
export LD_LIBRARY_PATH="$(dirname $LIBSTDCXX_SO):$LD_LIBRARY_PATH"
export LD_PRELOAD="$LIBSTDCXX_SO"
```

### CUDA architecture mismatch

Set the correct CUDA architecture for your GPU:

```bash
# For GH200 (SM90)
export TORCH_CUDA_ARCH_LIST="9.0;9.0a"
export CUDAARCHS="90;90a"

# For other GPUs, check your architecture:
nvidia-smi --query-gpu=compute_cap --format=csv
```

### Flash Attention build failures on ARM64

Use single-threaded builds:

```bash
export MAX_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1
export NVCC_THREADS=1
```

### Triton import errors

Ensure Triton version is compatible with PyTorch:

```bash
# For PyTorch 2.5.x, use Triton 3.1.0
pip uninstall triton
pip install triton==3.1.0
```

## Running Benchmarks

See the `scripts/` directory for SLURM job scripts:

```bash
# Run nano-vllm benchmark
sbatch scripts/bench.slurm

# Run vLLM benchmark
sbatch scripts/bench_vllm.slurm

# Compare results
bash scripts/compare_results.sh
```
