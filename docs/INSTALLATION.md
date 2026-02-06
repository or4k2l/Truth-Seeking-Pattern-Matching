# Installation Guide

## Prerequisites

- Python 3.9 or higher
- CUDA 12.x (for GPU support)
- At least 8GB of RAM
- 10GB of free disk space

## Quick Installation

### 1. Clone the Repository

```bash
git clone https://github.com/or4k2l/robust-vision.git
cd robust-vision
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Package

```bash
pip install -e .
```

## GPU Setup

### CUDA 12.x

The default installation includes JAX with CUDA 12 support:

```bash
pip install "jax[cuda12]>=0.4.20"
```

### CUDA 11.x

If you need CUDA 11 support:

```bash
pip install "jax[cuda11_cudnn82]>=0.4.20"
```

### CPU Only

For CPU-only installation:

```bash
pip install jax>=0.4.20
```

## Docker Installation

### Build Docker Image

```bash
docker build -t robust-vision:latest .
```

### Run Training in Docker

```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  robust-vision:latest
```

### Interactive Docker Session

```bash
docker run --gpus all -it \
  -v $(pwd):/app \
  robust-vision:latest bash
```

## Verification

Verify your installation:

```python
import jax
import flax
import tensorflow as tf

print(f"JAX version: {jax.__version__}")
print(f"Flax version: {flax.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"JAX devices: {jax.devices()}")
```

## Troubleshooting

### CUDA Not Found

If JAX doesn't detect your GPU:

1. Check CUDA installation: `nvcc --version`
2. Reinstall JAX with CUDA support
3. Check CUDA driver compatibility

### Out of Memory

If you encounter OOM errors:

1. Reduce batch size in config
2. Use gradient accumulation
3. Enable mixed precision training

### TensorFlow Warnings

TensorFlow warnings about GPU can be ignored if you're only using it for data loading.

## Cloud Setup

### Google Colab

```python
!pip install git+https://github.com/or4k2l/robust-vision.git
```

### AWS

Use Deep Learning AMI with CUDA 12:

```bash
aws ec2 run-instances \
  --image-id ami-xxxxxxxxx \
  --instance-type p3.2xlarge \
  --key-name your-key
```

### GCP

Use Deep Learning VM Image:

```bash
gcloud compute instances create robust-vision \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --image-family=common-cu121 \
  --image-project=deeplearning-platform-release
```

## Next Steps

- Read [TRAINING.md](TRAINING.md) for training instructions
- Check [DEPLOYMENT.md](DEPLOYMENT.md) for deployment options
- Explore example notebooks in `notebooks/`
