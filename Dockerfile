# Use an official PyTorch image with CUDA support as the base
# This is larger but guarantees GPU compatibility
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies for audio processing (librosa needs these)
# We don't need gcc here as often because pre-built wheels usually exist, 
# but good to have just in case.
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install python dependencies
# --extra-index-url ensures we look for cpu-only versions if needed, 
# but the base image already has gpu-torch. 
# Pip should handle this gracefully and mostly install the other libs.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]