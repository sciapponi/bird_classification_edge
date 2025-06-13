#!/bin/bash

# --- Configuration - MODIFICA QUESTE VARIABILI SE NECESSARIO ---
IMAGE_NAME="bird_classification_edge"
CONTAINER_NAME_DEFAULT="bird_train_container" # Nome container di default se non specificato

# Percorsi relativi alla posizione dello script (presumendo che i dataset siano nella root del progetto)
HOST_BIRD_SOUND_DATASET_DIR="$PWD/bird_sound_dataset"
HOST_ESC50_DIR="$PWD/esc-50/ESC-50-master"
HOST_AUGMENTED_DATASET_DIR="$PWD/augmented_dataset" # Lascia vuoto "" se non usi pregenerated_no_birds o la cartella non esiste
HOST_CONFIG_DIR="$PWD/config"  # Presume che la cartella config sia nella stessa directory dello script
HOST_LOGS_DIR="$PWD/logs"      # Presume che la cartella logs sia nella stessa directory dello script

# Percorsi all'interno del container (generalmente non serve modificarli se il Dockerfile usa /app)
CONTAINER_APP_DIR="/app"
CONTAINER_BIRD_SOUND_DATASET_DIR="$CONTAINER_APP_DIR/bird_sound_dataset"
CONTAINER_ESC50_DIR="$CONTAINER_APP_DIR/ESC-50-master"
CONTAINER_AUGMENTED_DATASET_DIR="$CONTAINER_APP_DIR/augmented_dataset"
CONTAINER_CONFIG_DIR="$CONTAINER_APP_DIR/config"
CONTAINER_LOGS_DIR="$CONTAINER_APP_DIR/logs"

# --- Fine Configurazione ---

# Gestione argomenti
CONTAINER_NAME="$CONTAINER_NAME_DEFAULT"
GPU_ID=""
EXTRA_HYDRA_PARAMS=""
RUN_ON_MAC=false # Flag per esecuzione su Mac

# Parsing degli argomenti per nome container, GPU e parametri Hydra
# Esempio: ./run_docker_training.sh mio_container GPU_ID=0 training.epochs=10
# Per Mac: ./run_docker_training.sh mio_container MAC training.epochs=10
if [ "$#" -ge 1 ]; then
    CONTAINER_NAME=$1
    shift # Rimuove il primo argomento (nome container)
    while (( "$#" )); do
        if [[ "$1" == "GPU_ID="* ]]; then
            GPU_ID="${1#GPU_ID=}"
        elif [[ "$1" == "MAC" ]]; then # Aggiunto check per flag MAC
            RUN_ON_MAC=true
        else
            EXTRA_HYDRA_PARAMS="$EXTRA_HYDRA_PARAMS $1"
        fi
        shift
    done
fi

echo "--- Avvio container Docker ---"
echo "Nome Container: $CONTAINER_NAME"

# Rinomino RUN_ON_MAC in USE_GPU per coerenza
USE_GPU=true
if [ "$RUN_ON_MAC" = true ]; then
    USE_GPU=false
fi

# GPU configuration
if [[ $USE_GPU == true ]]; then
    if [[ -n $GPU_ID ]]; then
        echo "GPU ID: $GPU_ID"
        GPU_FLAG_OPTS="--gpus device=$GPU_ID"
    else
        echo "GPU ID: all (default)"
        GPU_FLAG_OPTS="--gpus all"
    fi
else
    echo "GPU ID: N/A (Esecuzione su Mac, CPU forzata)"
    GPU_FLAG_OPTS=""
fi

echo "Parametri Hydra aggiuntivi: ${EXTRA_HYDRA_PARAMS:-Nessuno}"
echo "-----------------------------"

# Costruzione dei mount per i volumi
VOLUME_MOUNTS=""

# Verifica l'esistenza delle directory prima di aggiungere i mount
if [ -d "$HOST_BIRD_SOUND_DATASET_DIR" ]; then
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v $HOST_BIRD_SOUND_DATASET_DIR:$CONTAINER_BIRD_SOUND_DATASET_DIR"
else
    echo "Attenzione: Directory bird_sound_dataset non trovata in $HOST_BIRD_SOUND_DATASET_DIR"
fi

if [ -d "$HOST_ESC50_DIR" ]; then
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v $HOST_ESC50_DIR:$CONTAINER_ESC50_DIR"
else
    echo "Attenzione: Directory ESC-50-master non trovata in $HOST_ESC50_DIR"
fi

if [ -n "$HOST_AUGMENTED_DATASET_DIR" ] && [ -d "$HOST_AUGMENTED_DATASET_DIR" ]; then
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v $HOST_AUGMENTED_DATASET_DIR:$CONTAINER_AUGMENTED_DATASET_DIR"
elif [ -n "$HOST_AUGMENTED_DATASET_DIR" ]; then # Se la variabile è settata ma la dir non esiste
    echo "Attenzione: Directory augmented_dataset non trovata in $HOST_AUGMENTED_DATASET_DIR"
fi

if [ -d "$HOST_CONFIG_DIR" ]; then
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v $HOST_CONFIG_DIR:$CONTAINER_CONFIG_DIR"
else
    echo "ERRORE CRITICO: Directory config non trovata in $HOST_CONFIG_DIR. Impossibile continuare."
    exit 1
fi

# Crea la directory dei log sull'host se non esiste
if [ ! -d "$HOST_LOGS_DIR" ]; then
    echo "Creazione directory logs in $HOST_LOGS_DIR"
    mkdir -p "$HOST_LOGS_DIR"
fi
VOLUME_MOUNTS="$VOLUME_MOUNTS -v $HOST_LOGS_DIR:$CONTAINER_LOGS_DIR"


# Comando Docker run
# shellcheck disable=SC2086
docker run $GPU_FLAG_OPTS --name "$CONTAINER_NAME" --rm -it --shm-size=16gb \
    $VOLUME_MOUNTS \
    "$IMAGE_NAME" \
    python train.py $EXTRA_HYDRA_PARAMS

echo "--- Esecuzione container Docker terminata ---" 