# Robust Vision Production Image
# Repository: https://github.com/or4k2l/robust-vision

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

LABEL maintainer="or4k2l"
LABEL repository="https://github.com/or4k2l/robust-vision"
LABEL description="Production-ready robust vision training framework"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV JAX_PLATFORM_NAME=cuda

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . /app/

# Install the package
RUN pip install -e .

# Create directories for data, checkpoints, and logs
RUN mkdir -p /app/data /app/checkpoints /app/logs /app/results

# Set the default command
CMD ["python", "scripts/train.py", "--config", "configs/baseline.yaml"]
