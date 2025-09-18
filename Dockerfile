FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev build-essential \
    git ffmpeg curl ca-certificates libgl1 libglib2.0-0 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy minimal first (to leverage Docker cache)
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt && \
    # PyTorch CUDA 11.8 (per MuseTalk README recommends 2.0.1)
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
      --index-url https://download.pytorch.org/whl/cu118 && \
    # FishAudio SDK
    pip install fish-audio-sdk && \
    # MuseTalk deps
    pip install transformers==4.39.2 accelerate==0.28.0 diffusers==0.30.2 \
      huggingface_hub==0.30.2 librosa==0.11.0 einops==0.8.1 gradio==5.24.0 \
      omegaconf ffmpeg-python moviepy tqdm && \
    # MMLab stack (pin + prebuilt wheel for CUDA 11.8 + torch2.0.1)
    pip install mmengine==0.10.7 && \
    pip install -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.1/index.html mmcv==2.0.1 && \
    pip install mmdet==3.1.0 mmpose==1.1.0

# Copy the full repo
COPY . /app

# Clone MuseTalk and fetch weights
RUN chmod +x /app/scripts/setup_musetalk.sh && \
    /app/scripts/setup_musetalk.sh && \
    pip install -U "huggingface_hub[cli]" gdown && \
    bash /app/external/MuseTalk/download_weights.sh

# Default working dir
WORKDIR /app

# Default command shows help; override in docker-compose or docker run
CMD ["python", "-m", "src.live_avatar.main", "--help"]
