# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install libgomp1 for PyTorch/Torchaudio compatibility
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first and install packages
# This benefits from Docker layer caching if requirements don't change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Non è più necessario creare utenti/gruppi o cambiare USER
# I comandi verranno eseguiti come root (o l'utente predefinito dell'immagine base)

# Set the default command to run when the container starts
CMD ["python", "train.py"] 