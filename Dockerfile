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

# Copy the requirements file first and install packages
# This benefits from Docker layer caching if requirements don't change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs && chmod -R 777 /app/logs
RUN mkdir -p /app/benchmark/benchmark_results && chmod -R 777 /app/benchmark/benchmark_results

# Set environment variables for audio processing
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV MATPLOTLIB_BACKEND="Agg"

# Non è più necessario creare utenti/gruppi o cambiare USER
# I comandi verranno eseguiti come root (o l'utente predefinito dell'immagine base)

# Set the default command to run when the container starts
# Default to training, but can be overridden for benchmarking
CMD ["python", "train.py"] 