# Installation

## Environment

The pinned dependencies target Linux x86_64, Python 3.12, and CUDA 12.8 on NVIDIA GPUs.
Use driver version 570 or newer.

```bash
git clone https://github.com/xiaoqian-shen/AVTR.git
cd AVTR
conda create -n avtr python=3.12 -y
conda activate avtr
pip install -r requirements.txt
```

The requirements include training, inference, RL, and evaluation dependencies.
Versions are pinned in [requirements.txt](../requirements.txt).
Run the documented commands from the repository root.

Hugging Face model repositories are downloaded through the included `huggingface_hub` package.
Set `HF_TOKEN` for private models or gated datasets and `HF_HOME` to choose the cache directory.

## Video decoding

Video inputs use torchcodec and require FFmpeg 4–7 shared libraries:

```bash
conda install -c conda-forge "ffmpeg>=4,<8" -y
```

## Optional CUDA kernels

FlashAttention is optional for stages 0–2, which fall back to PyTorch SDPA, and required for RL:

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

The CUDA toolkit is needed to build it. For faster GatedDeltaNet short convolutions, also install:

```bash
pip install causal-conv1d --no-build-isolation
```
