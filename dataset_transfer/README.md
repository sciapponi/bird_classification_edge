# Dataset Transfer Tools

Strumenti per l'analisi, preprocessing e trasferimento di dataset audio per il progetto di classificazione uccelli.

## 📊 Dataset Analyzer Realistico

Lo script `dataset_analyzer_realistic.py` analizza dataset audio con **benchmark realistico** che simula il preprocessing identico a `train_distillation.py` per stime accurate.

### Utilizzo Base

```bash
# Analisi completa con benchmark realistico
python dataset_analyzer_realistic.py /path/to/your/dataset

# Esempio con il dataset del progetto
python dataset_analyzer_realistic.py ../bird_sound_dataset

# Personalizza numero di file per benchmark (default: 30)
python dataset_analyzer_realistic.py ../bird_sound_dataset --benchmark-samples 20

# Analizza solo la directory principale (senza sottodirectory)
python dataset_analyzer_realistic.py /path/to/dataset --no-subdirs

# Limita la profondità di scansione
python dataset_analyzer_realistic.py /path/to/dataset --max-depth 2
```

### 🧪 Benchmark Realistico

Lo script esegue un **benchmark automatico** su un campione di file audio per simulare il preprocessing completo di `train_distillation.py`:

#### Fasi del Preprocessing Simulate:
1. **📁 Caricamento Audio**: `torchaudio.load()` con gestione errori
2. **🔧 Preprocessing**: 
   - Resampling a 32kHz
   - Conversione mono
   - Estrazione chiamate (`extract_call_segments`)
   - Normalizzazione
   - Augmentazioni audio (50% probabilità)
3. **📊 Generazione Spettrogrammi**:
   - Mel spectrogram (64 bins)
   - Linear spectrogram (64 bins)  
   - n_fft=1024, hop_length=320
   - ~298 frames per file (3.0s @ 32kHz)
4. **💾 Salvataggio**: Formato NPZ compresso

#### Parametri Identici al Progetto:
- **Sample rate**: 32000 Hz
- **Durata clip**: 3.0 secondi (96,000 campioni)
- **Spettrogramma**: combined_log_linear con filtri apprendibili
- **Output**: File .npz con metadati completi

### Output dell'Analisi

Lo script fornisce **6 sezioni** di analisi dettagliata:

#### 1. 📊 STATISTICHE GENERALI DATASET
- Numero totale di file e dimensione
- File audio vs altri file
- Dimensione media, mediana, min e max

#### 2. 🧪 RISULTATI BENCHMARK REALISTICO
- **Tempi realistici** per ogni fase del preprocessing
- **Breakdown dettagliato**: Load → Preprocess → Spectrograms → Save
- **Tasso di successo** del processing
- **Compressione effettiva** misurata

#### 3. 🎯 STIME REALISTICHE
- **Tempo totale** preprocessing con margine sicurezza +30%
- **Dimensioni output** basate su compressione misurata
- **Breakdown tempo** per ogni fase su tutto il dataset
- **Storage requirements** (temporaneo + finale)

#### 4. 📡 STIME TRASFERIMENTO
- **Multiple velocità**: 5, 10, 25, 50, 100 Mbps
- **Confronto tempi**: Dataset originale vs processato
- **Risparmio tempo** effettivo

#### 5. 💡 RACCOMANDAZIONI SPECIFICHE
- **Strategie processing**: Parallelo, batch, checkpointing
- **Ottimizzazioni storage**: NPZ vs HDF5/Parquet
- **Timeline suggerita**: Esecuzione notturna, background

#### 6. 📈 DETTAGLI TECNICI
- **Frame per file**, campioni audio
- **Parametri spettrogrammi** utilizzati
- **Metadati** salvati nel preprocessing

### Esempio Output Realistico

```
🚀 ANALISI DATASET CON BENCHMARK REALISTICO
🔬 Simula preprocessing identico a train_distillation.py
================================================================================

📋 Trovati 4,449 file da analizzare...
✅ Scansione completata in 2.3s
🎵 File audio: 4,301

🧪 BENCHMARK PREPROCESSING REALISTICO
📊 Campione: 30 file
🔬 Simula: train_distillation.py pipeline completa

✅ Benchmark realistico completato!
📊 Campioni processati: 30/30 (100.0%)
⏱️  Tempo TOTALE medio: 1.42s ± 0.23s

🔍 BREAKDOWN TEMPI:
   📁 Caricamento: 0.067s (4.7%)
   🔧 Preprocessing: 0.431s (30.4%)
   📊 Spettrogrammi: 0.080s (5.6%)
   💾 Salvataggio: 0.842s (59.3%)

📦 COMPRESSIONE REALISTICA:
   Rapporto medio: 18.3% ± 4.2%

🎯 STIME REALISTICHE (basate su train_distillation.py)
================================================================================
🎵 File audio da processare: 4,301
⏱️  Tempo stimato preprocessing: 2.7h
   (Include margine sicurezza +30% per overhead sistema)
💾 Dimensione input: 12.84 GB
💾 Dimensione stimata output: 2.35 GB
📉 Compressione: 18.3%

⏰ BREAKDOWN TEMPO TOTALE:
   📁 Caricamento: 4.8m
   🔧 Preprocessing: 30.9m
   📊 Spettrogrammi: 5.7m
   💾 Salvataggio: 1.0h

📡 STIME TRASFERIMENTO:
Velocità    Dataset Originale  Dataset Processato   Risparmio
----------------------------------------------------------------------------
  5 Mbps     4.6h               42.1m                3.9h
 10 Mbps     2.3h               21.0m                1.9h
 25 Mbps     55.1m              8.4m                 46.7m
 50 Mbps     27.5m              4.2m                 23.3m
100 Mbps     13.8m              2.1m                 11.7m

💡 RACCOMANDAZIONI SPECIFICHE:
   ⚡ Preprocessing lungo (2.7h)
       → Esegui durante la notte o in background
   ✅ Ottima compressione (18.3%)

💽 STORAGE REQUIREMENTS:
   📦 Storage temporaneo necessario: 3.64 GB
   📀 Storage finale (dopo cleanup): 2.35 GB
```

### Formati Audio Supportati

```
.wav, .mp3, .flac, .ogg, .m4a, .aac, .wma, 
.aiff, .au, .3gp, .amr, .opus
```

### Parametri Avanzati

```bash
# Numero campioni per benchmark (più campioni = stime più accurate)
--benchmark-samples N     # Default: 30

# Controllo scansione directory
--no-subdirs              # Solo directory principale
--max-depth N             # Profondità massima scansione

# Esempi:
python dataset_analyzer_realistic.py ../bird_sound_dataset --benchmark-samples 50
python dataset_analyzer_realistic.py /dataset --no-subdirs --benchmark-samples 10
```

### 🔧 Dipendenze

Lo script usa **solo librerie standard Python** - nessuna installazione aggiuntiva richiesta!

Dipendenze incluse: `pathlib`, `json`, `collections`, `argparse`, `time`, `numpy`

### 📝 Caratteristiche Chiave

- **🎯 Stime accurate**: Benchmark su preprocessing reale
- **⚡ Veloce**: Scansione ottimizzata per dataset grandi (500GB+)
- **🔍 Dettagliato**: Breakdown completo di ogni fase
- **💾 Gestione memoria**: Cleanup automatico file temporanei
- **📊 Multi-velocità**: Stime trasferimento per diverse connessioni
- **🛡️ Robusto**: Gestione errori e file corrotti
- **📈 Progressivo**: Progress tracking in tempo reale

### 🧪 Accuratezza del Benchmark

Il benchmark simula **fedelmente** il preprocessing di `train_distillation.py`:

- ✅ **Timing realistico** per ogni operazione
- ✅ **Parametri identici** (32kHz, 3.0s, n_fft=1024, etc.)
- ✅ **Formato output** NPZ compresso 
- ✅ **Metadati completi** come nel progetto
- ✅ **Margine sicurezza** per overhead sistema

### 🚀 Utilizzo Raccomandato

1. **Analizza il dataset** per capire dimensioni e complessità
2. **Pianifica preprocessing** basandoti sui tempi realistici  
3. **Prepara storage** secondo i requirements calcolati
4. **Esegui preprocessing** con confidence sui tempi stimati
5. **Trasferisci dataset** ottimizzato al server 