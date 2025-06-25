# Use an official Python runtime with build tools
FROM python:3.9

# Install required system dependencies for audio processing, compilation, and BirdNET
RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential \
    ffmpeg \
    libsndfile1-dev \
    libasound2-dev \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Configure pip for better timeout handling
RUN pip config set global.timeout 300
RUN pip config set global.retries 3

# Install PyTorch with CUDA support first
RUN pip install --no-cache-dir --timeout 300 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy the requirements file and install other packages
COPY requirements.txt .
# Remove only exact torch package lines from requirements to avoid conflicts
RUN sed -i '/^torch$/d; /^torchaudio$/d; /^torchvision$/d' requirements.txt

# Install packages with retry logic for birdnetlib specifically
RUN pip install --no-cache-dir --timeout 300 \
    numpy pandas resampy matplotlib seaborn scikit-learn \
    hydra-core omegaconf tqdm librosa torchsummary

# Install birdnetlib separately with multiple retries
RUN for i in 1 2 3; do \
    pip install --no-cache-dir --timeout 600 birdnetlib && break || \
    (echo "Attempt $i failed, retrying..." && sleep 10); \
    done

# Copy the rest of the application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs && chmod -R 777 /app/logs
RUN mkdir -p /app/benchmark/benchmark_results && chmod -R 777 /app/benchmark/benchmark_results

# Set environment variables for audio processing
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV MATPLOTLIB_BACKEND="Agg"
ENV PYTHONIOENCODING=UTF-8

# Non è più necessario creare utenti/gruppi o cambiare USER
# I comandi verranno eseguiti come root (o l'utente predefinito dell'immagine base)

# Set the default command to run when the container starts
# Default to training, but can be overridden for benchmarking
CMD ["python", "train.py"] 