# 🐦 Bird Classification Benchmark System

Sistema completo per confrontare le prestazioni del tuo modello di classificazione degli uccelli con BirdNET come riferimento.

## 🚀 Quick Start

```bash
# Dal directory root del progetto
cd benchmark
source ../venv/bin/activate

# Run standard benchmark
python run_benchmark.py --config-name=quick_start

# Test new features (quick validation)
python run_benchmark.py --config-name=test_benchmark

# Birds-only benchmark (no no_birds class)
python run_benchmark.py --config-name=birds_only_benchmark

# Adaptive threshold benchmark (reduce false positives)
python run_benchmark.py --config-name=adaptive_threshold_benchmark

# Statistical analysis of dataset requirements
python run_benchmark.py --config-name=statistical_analysis

# Multiple runs for confidence intervals
python run_benchmark.py --config-name=multiple_runs_benchmark
```

## 🐳 Docker Usage

Il sistema supporta Docker per esecuzione isolata e riproducibile:

```bash
# Sintassi generale
./run_docker_benchmark.sh <container_name> <gpu_id> <config_name>

# Esempi pratici
./run_docker_benchmark.sh my_benchmark 0 quick_start          # GPU 0
./run_docker_benchmark.sh cpu_test cpu adaptive_threshold    # Solo CPU
./run_docker_benchmark.sh gpu1_birds 1 birds_only_benchmark  # GPU 1
```

**Parametri Docker:**
- `container_name`: Nome univoco per il container
- `gpu_id`: ID GPU (0, 1, 2...) o "cpu" per esecuzione CPU-only
- `config_name`: Nome della configurazione (senza .yaml)

Il sistema rileva automaticamente la presenza di GPU e usa CPU come fallback su Mac/sistemi senza NVIDIA GPU.

## 📋 Configurazioni Benchmark Disponibili

Il sistema offre **9 configurazioni specializzate** per diversi tipi di analisi e testing. Ecco la panoramica completa:

### 📊 **Tabella Comparativa Configurazioni**

| Configurazione | Soglia BirdNET | Soglia Student | File Limit | Force Bird | Adaptive | Multiple Runs | Scopo Principale |
|----------------|----------------|----------------|------------|------------|----------|---------------|------------------|
| **`benchmark`** | 0.2 | 0.1 | Tutti | ❌ | ❌ | ❌ | Standard completo |
| **`mini_benchmark`** | 0.25 | 0.1 | 15 | ❌ | ❌ | ❌ | Pipeline completa veloce |
| **`test_benchmark`** | 0.25 | 0.1 | 30 | ❌ | ❌ | ❌ | Validazione sviluppo |
| **`quick_start`** | 0.1 | 0.1 | 3 | ❌ | ❌ | ❌ | Primo utilizzo |
| **`birds_only_benchmark`** | 0.50 | 0.05 | Tutti | ✅ | ❌ | ❌ | Solo specie (no no_birds) |
| **`optimized_benchmark`** | 0.18 | 0.05 | Tutti | ❌ | ❌ | ❌ | Massima precisione |
| **`adaptive_threshold_benchmark`** | 0.25+0.4 | 0.05 | Tutti | ❌ | ✅ | ❌ | Ridurre falsi positivi |
| **`multiple_runs_benchmark`** | 0.25+0.4 | 0.1 | 200 | ❌ | ✅ | ✅ (5x) | Validazione statistica |
| **`statistical_analysis`** | N/A | N/A | Tutti | ❌ | ❌ | ❌ | Solo analisi dataset |

### 🚀 **Comandi di Avvio Rapido**

```bash
# Test velocissimo (3 file)
./run_docker_benchmark.sh quick 0 quick_start

# Test completo veloce (15 file) 
./run_docker_benchmark.sh mini 0 mini_benchmark

# Standard completo
./run_docker_benchmark.sh standard 0 benchmark

# Solo uccelli (mai no_birds) 🔥 NOVITÀ
./run_docker_benchmark.sh birds_only 0 birds_only_benchmark

# Soglie ottimizzate
./run_docker_benchmark.sh optimized 0 optimized_benchmark

# Validazione statistica
./run_docker_benchmark.sh multi_runs 0 multiple_runs_benchmark
```

### 🎯 **Scegli la Configurazione Giusta**

**🚀 Per iniziare velocemente:**
- `quick_start` - Primo test (3 file, 30 secondi)
- `mini_benchmark` - Pipeline completa veloce (15 file, 2 minuti)

**🔬 Per test e sviluppo:**
- `test_benchmark` - Validazione funzionalità (30 file)
- `benchmark` - Standard di riferimento

**🎯 Per analisi specializzate:**
- `birds_only_benchmark` - **Confronto equo solo uccelli** (esclude no_birds)
- `optimized_benchmark` - Configurazione ottimizzata per massima accuratezza
- `adaptive_threshold_benchmark` - Riduce falsi positivi no_birds

**📊 Per validazione scientifica:**
- `multiple_runs_benchmark` - 5 run con confidence intervals
- `statistical_analysis` - Sample size e power analysis

### 🔧 1. `benchmark.yaml` - Configurazione Base

**Cosa fa:**
Configurazione standard completa che confronta Student model vs BirdNET su tutte le 9 classi (8 specie + no_birds).

**Preprocessing:**
- **Sample rate**: 32kHz (identico al training)
- **Durata segmenti**: 3.0 secondi
- **Bandpass filter**: 150-16000 Hz (Butterworth order 4)
- **Extract calls**: `true` - estrazione automatica di chiamate di uccelli
- **Fallback**: clip random 3s se nessuna chiamata rilevata
- **Normalizzazione**: amplitude normalization a [-1, 1]

**Scelte implementative:**
- **Fair comparison**: Preprocessing identico per entrambi i modelli
- **Species filtering**: BirdNET limitato alle 8 specie target + gestione no_birds
- **Confidence thresholds**: Student 0.1, BirdNET 0.25/0.4 (adaptive)
- **No data leakage**: Usa dataset esistente senza duplicazioni

**Parametri configurabili:**
```yaml
birdnet:
  confidence_threshold: 0.25      # Soglia base BirdNET
  no_birds_threshold: 0.4         # Soglia separata per no_birds
  use_adaptive_threshold: true    # Abilita soglie adattive
  
student_model:
  inference:
    confidence_threshold: 0.1     # Soglia Student model
    batch_size: 32               # Dimensione batch
```

**Come avviarlo:**
```bash
# Python locale
python run_benchmark.py --config-name=benchmark

# Docker (GPU 0)
./run_docker_benchmark.sh standard_bench 0 benchmark

# Docker (CPU-only)
./run_docker_benchmark.sh cpu_bench cpu benchmark
```

---

### ⚡ 2. `quick_start.yaml` - Avvio Rapido

**Cosa fa:**
Versione semplificata della configurazione base per test rapidi e getting started.

**Preprocessing:**
Identico alla configurazione base ma con impostazioni semplificate per facilità d'uso.

**Scelte implementative:**
- **Configurazione minimale**: Solo parametri essenziali
- **Debug mode opzionale**: `dev_mode: true` per test con subset
- **Auto-detection**: Rileva automaticamente paths e configurazioni
- **Log semplificati**: Livello INFO per output pulito

**Parametri configurabili:**
```yaml
debug:
  dev_mode: true              # Modalità sviluppo
  test_with_subset: true      # Test solo con subset
  subset_size: 3              # File per classe nel subset
  files_limit: null           # Limite totale file (null = tutti)

segmentation:
  duration: 3.0               # Durata segmenti

logging:
  level: "INFO"               # Livello di log
```

**Come avviarlo:**
```bash
# Test rapido con subset (3 file per classe)
python run_benchmark.py --config-name=quick_start debug.dev_mode=true

# Benchmark completo ma semplificato
python run_benchmark.py --config-name=quick_start debug.dev_mode=false

# Docker
./run_docker_benchmark.sh quick_test 0 quick_start
```

---

### 🧪 3. `test_benchmark.yaml` - Test Mode

**Cosa fa:**
Configurazione per validazione rapida di nuove funzionalità e debugging del sistema.

**Preprocessing:**
Identico alle altre configurazioni ma limitato a sample molto piccoli.

**Scelte implementative:**
- **Validation-only**: Verifica che tutto funzioni senza analisi completa
- **Files limitati**: Solo 30 files totali (5 per classe)
- **Test thresholds**: Soglie ottimizzate per test rapidi
- **Initialization check**: Verifica caricamento modelli e configurazioni

**Parametri configurabili:**
```yaml
test_mode: true               # Abilita modalità test globale

birdnet:
  confidence_threshold: 0.3   # Soglia più alta per test
  no_birds_threshold: 0.45    # Soglia no_birds per test
  adaptive_factor: 1.5        # Fattore adattivo

debug:
  files_limit: 30             # Massimo 30 file
  max_files_per_class: 5      # Massimo 5 file per classe
```

**Come avviarlo:**
```bash
# Test validazione sistema
python run_benchmark.py --config-name=test_benchmark

# Docker (ottimo per CI/CD)
./run_docker_benchmark.sh system_test cpu test_benchmark
```

---

### 🎯 4. `mini_benchmark.yaml` - Mini Pipeline Completo

**Cosa fa:**
Esegue la **pipeline completa** (discovery → prediction → comparison) ma con dataset ridottissimo per test approfonditi.

**Preprocessing:**
Pipeline identica al benchmark completo con tutti i passaggi di preprocessing.

**Scelte implementative:**
- **Complete pipeline**: Tutti i passaggi inclusi (a differenza di test_benchmark)
- **Minimal dataset**: 15 files totali (3 per classe)
- **Full analysis**: Genera tutti gli output (CSV, JSON, PNG)
- **Real comparison**: Confronto vero ma veloce

**Parametri configurabili:**
```yaml
benchmark:
  name: "mini_benchmark_test"   # Nome specifico

debug:
  files_limit: 15              # Totale 15 file
  max_files_per_class: 3       # 3 file per classe

birdnet:
  confidence_threshold: 0.3    # Soglie ottimizzate
  no_birds_threshold: 0.45
  use_adaptive_threshold: true
```

**Come avviarlo:**
```bash
# Mini benchmark completo
python run_benchmark.py --config-name=mini_benchmark

# Docker (perfetto per validazione pre-produzione)
./run_docker_benchmark.sh mini_test 0 mini_benchmark
```

---

### 🦅 5. `birds_only_benchmark.yaml` - Solo Specie di Uccelli

**Cosa fa:**
Confronto **fair** tra modelli escludendo completamente la classe `no_birds` per focus solo sulle 8 specie.

**Preprocessing:**
Identico ma con **esclusione completa** dei campioni no_birds dal test set.

**Scelte implementative:**
- **8-class comparison**: Solo specie di uccelli (Bubo_bubo, Certhia_familiaris, etc.)
- **No no_birds bias**: Elimina la complessità della classe artificiale
- **Lower BirdNET threshold**: 0.15 invece di 0.25 per catturare più specie
- **Species-focused metrics**: Metriche ottimizzate per classificazione specie

**Parametri configurabili:**
```yaml
benchmark:
  mode:
    birds_only: true                    # Modalità solo-uccelli
    exclude_no_birds_from_ground_truth: true  # Rimuovi no_birds dal dataset
    force_fair_preprocessing: true      # Preprocessing identico

birdnet:
  confidence_threshold: 0.15           # Soglia più bassa per specie
  no_birds_threshold: 0.6             # Soglia alta per evitare false no_birds
  use_adaptive_threshold: false       # Disabilitato in modalità birds-only

comparison:
  metrics:
    exclude_classes: ["no_birds"]      # Escludi no_birds dalle metriche
```

**Come avviarlo:**
```bash
# Confronto solo specie
python run_benchmark.py --config-name=birds_only_benchmark

# Docker
./run_docker_benchmark.sh birds_only 0 birds_only_benchmark
```

---

### 🎯 6. `adaptive_threshold_benchmark.yaml` - Soglie Adattive

**Cosa fa:**
Ottimizza le soglie di confidenza per **ridurre i falsi positivi** della classe no_birds mantenendo buone performance sulle specie.

**Preprocessing:**
Identico alle altre configurazioni con focus sulla gestione delle soglie di decisione.

**Scelte implementative:**
- **Dual thresholds**: Soglia base (0.25) + soglia no_birds (0.4)
- **Adaptive logic**: Logica per evitare no_birds quando confidence è intermedia
- **False positive reduction**: Specificamente progettato per ridurre no_birds errati
- **Enhanced metrics**: Metriche aggiuntive per analisi delle soglie

**Parametri configurabili:**
```yaml
birdnet:
  confidence_threshold: 0.25          # Soglia base più alta
  no_birds_threshold: 0.4             # Soglia dedicata no_birds
  use_adaptive_threshold: true        # Sistema soglie adattive
  adaptive_factor: 1.6               # Moltiplicatore per decisioni no_birds
  species_bias_factor: 1.1           # Bias verso specie vs no_birds
  min_detection_duration: 0.5        # Durata minima detection valida

comparison:
  metrics:
    calculate_no_birds_metrics: true   # Metriche specifiche no_birds
    calculate_species_vs_no_birds_confusion: true  # Confusione specie/no_birds
    
  visualization:
    threshold_analysis:
      confidence_distribution: true    # Distribuzione confidence
      roc_curve: true                 # Curva ROC
      precision_recall_curve: true    # Curva PR
```

**Come avviarlo:**
```bash
# Benchmark con soglie ottimizzate
python run_benchmark.py --config-name=adaptive_threshold_benchmark

# Docker
./run_docker_benchmark.sh adaptive_test 0 adaptive_threshold_benchmark
```

---

### 📊 7. `statistical_analysis.yaml` - Analisi Statistica

**Cosa fa:**
**Non esegue benchmark** ma analizza il dataset per calcolare sample size, power analysis e validazione statistica.

**Preprocessing:**
Nessun preprocessing audio - solo **scansione dataset** per conteggio campioni e analisi distribuzione classi.

**Scelte implementative:**
- **Dataset scanning**: Conta files per classe senza caricarli
- **Sample size calculation**: Calcola campioni necessari per significatività statistica
- **Power analysis**: Determina potenza per rilevare differenze del 3-5%
- **Statistical validation**: Confidence intervals, effect size, t-test requirements

**Parametri configurabili:**
```yaml
statistical_analysis:
  confidence_level: 0.95              # Livello confidenza (95%)
  statistical_power: 0.80             # Potenza statistica (80%)
  minimum_effect_size: 0.05           # Minima differenza rilevabile (5%)
  recommended_runs: 3                 # Run consigliati per CI

analysis_type: "sample_size_calculation"  # Tipo di analisi
sample_size_only: true                # Solo calcolo sample size
```

**Output generato:**
```json
{
  "total_samples": 5076,
  "recommended_samples": 1591,
  "status": "ADEQUATE",
  "can_detect_improvement": true,
  "class_distribution": {...},
  "statistical_power": 0.80,
  "confidence_level": 0.95
}
```

**Come avviarlo:**
```bash
# Analisi statistica dataset
python run_benchmark.py --config-name=statistical_analysis

# Docker
./run_docker_benchmark.sh stats_analysis cpu statistical_analysis
```

---

### 📈 8. `multiple_runs_benchmark.yaml` - Multiple Run Statistici

**Cosa fa:**
Esegue **5 benchmark indipendenti** con sample casuali per calcolare confidence intervals e significance testing.

**Preprocessing:**
Preprocessing standard ma con **sample randomization** diversa per ogni run.

**Scelte implementative:**
- **Multiple independent runs**: 5 esecuzioni con semi random diversi
- **Fixed sample size**: 200 files (25 per classe) per consistency
- **Statistical aggregation**: Media, deviazione standard, confidence intervals
- **Paired t-test**: Test statistico per differenze significative
- **Effect size calculation**: Cohen's d per practical significance

**Parametri configurabili:**
```yaml
multiple_runs_mode: true              # Abilita modalità multiple runs
num_runs: 5                          # Numero di run da eseguire

statistical_validation:
  confidence_level: 0.95             # Livello confidenza CI
  minimum_detectable_difference: 0.03 # Differenza minima (3%)
  random_seed_base: 42               # Seed base (42, 43, 44...)

debug:
  files_limit: 200                   # Sample size fisso
  max_files_per_class: 25            # Bilanciato per classe
```

**Output generato:**
- **Individual runs**: Risultati di ogni singolo run
- **Aggregated results**: Media e confidence intervals
- **Statistical tests**: T-test, p-values, effect sizes
- **Comprehensive report**: Analisi statistica completa

**Come avviarlo:**
```bash
# Multiple runs per validazione statistica
python run_benchmark.py --config-name=multiple_runs_benchmark

# Docker (consigliato GPU per velocità)
./run_docker_benchmark.sh multi_runs 0 multiple_runs_benchmark
```

---

## 🎯 Cosa fa il Sistema

Il benchmark esegue automaticamente questi passi:

1. **📁 Audio Discovery**: Scopre tutti i file audio dal dataset esistente
2. **🤖 Student Model Predictions**: Genera predizioni con il tuo modello addestrato
3. **🦅 BirdNET Predictions**: Genera predizioni con BirdNET come riferimento
4. **📊 Model Comparison**: Confronta le prestazioni con metriche dettagliate

## 🆕 New Features & Improvements

### 🎯 Birds-Only Mode + Force Bird Prediction
Modalità completamente rinnovata che **forza entrambi i modelli a scegliere sempre una specie**:
```bash
./run_docker_benchmark.sh birds_only 0 birds_only_benchmark
# 🔄 FORCE BIRD PREDICTION: Both models will never predict no_birds
```
**Novità implementate:**
- ✅ **Force bird prediction**: Mai più predizioni "no_birds" 
- ✅ **Fair comparison**: Confronto equo solo su 8 specie
- ✅ **Intelligent fallback**: Gestione errori senza no_birds
- ✅ **Single threshold**: Sistema semplificato (0.50 BirdNET)

### 🔧 Simplified Threshold System
**BREAKING CHANGE**: Rimosso il complesso sistema dual-threshold in favore di un approccio più semplice e affidabile:
```bash
# Prima (complesso): confidence_threshold + no_birds_threshold
# Dopo (semplice): single confidence_threshold

./run_docker_benchmark.sh standard 0 benchmark  # Single threshold 0.2
```
**Miglioramenti:**
- ✅ **Single threshold**: Un solo parametro da configurare
- ✅ **Più prevedibile**: Comportamento deterministico
- ✅ **Facile debug**: Meno complessità, più affidabilità
- ✅ **Configurazioni aggiornate**: Tutti i YAML sistemati

### ⚡ Test Mode
Validazione rapida delle funzionalità con files limitati:
```bash
python run_benchmark.py --config-name=test_benchmark
# o con override:
python run_benchmark.py test_mode=true debug.files_limit=20
```

### 🔥 Force Bird Prediction (NOVITÀ)
Modalità che **forza entrambi i modelli a non predire mai "no_birds"**. Invece di no_birds, scelgono sempre la specie con confidence più alta.

**Quando è attivo:**
- ✅ **Birds-only mode**: `birds_only_benchmark` automaticamente attiva force bird prediction
- ✅ **Confronto equo**: Entrambi i modelli devono scegliere tra le 8 specie
- ✅ **Mai no_birds**: Nessun modello può "scappare" verso no_birds
- ✅ **Fallback intelligente**: In caso di errori, sceglie la prima specie target

**Esempio pratico:**
```bash
# Prima: BirdNET poteva predire "no_birds" per uccelli difficili
# Dopo: BirdNET DEVE scegliere la migliore tra le 8 specie

./run_docker_benchmark.sh birds_only 0 birds_only_benchmark
# Log: "🔄 FORCE BIRD PREDICTION: Both models will never predict no_birds"
```

**Implementazione tecnica:**
- `force_bird_prediction=True` nei costruttori dei predittori
- Logica modificata: `if confidence >= threshold: return best_species` 
- Gestione errori: Fallback a prima specie invece di no_birds

### 🔄 Fair Preprocessing
Entrambi i modelli (student e BirdNET) usano ora preprocessing identico:
- Segmenti 3s con extract_calls
- Sample rate 32kHz
- Bandpass filter 150-16000Hz
- Comparazione veramente equa

### 📊 Statistical Rigor
Analisi statistica completa per validazione scientifica:
```bash
# Analisi sample size e power analysis
python run_benchmark.py --config-name=statistical_analysis

# Multiple runs con confidence intervals
python run_benchmark.py --config-name=multiple_runs_benchmark
```

**Features statistiche:**
- **Sample size calculation**: Calcola campioni necessari per significatività
- **Power analysis**: Analizza potenza statistica per rilevare differenze
- **Multiple runs**: Confidence intervals e paired t-test
- **Effect size**: Cohen's d e practical significance
- **Report automatici**: Analisi statistica dettagliata

## 📊 Output e Risultati

Tutti i risultati vengono salvati in `benchmark/benchmark_results/YYYY-MM-DD_HH-MM-SS/`:

```
benchmark_results/
├── predictions/
│   ├── ground_truth.csv          # Ground truth auto-generato
│   ├── student_predictions.csv   # Predizioni del tuo modello
│   └── birdnet_predictions.csv   # Predizioni di BirdNET
└── comparison/
    ├── comparison_report.txt      # Report dettagliato
    ├── detailed_results.json     # Metriche complete
    ├── confusion_matrices.png    # Matrici di confusione
    ├── agreement_analysis.png    # Analisi di accordo tra modelli
    └── per_class_accuracy.png    # Accuratezza per classe
```

## ⚙️ Configurazione

### Quick Start (consigliato)
```yaml
# config/quick_start.yaml
benchmark:
  paths:
    audio_dir: "bird_sound_dataset"
    student_model: "best_distillation_model.pt"
    student_config: "config/bird_classification.yaml"
    output_dir: "benchmark_results"
```

### Configurazione Completa
```yaml
# config/benchmark.yaml - Configurazione avanzata con tutti i parametri
```

## 🛠️ Componenti del Sistema

### 1. Audio Discovery (`run_benchmark.py`)
- Scansiona automaticamente `bird_sound_dataset/` e `augmented_dataset/no_birds/`
- Genera ground truth dalla struttura delle cartelle
- Nessuna duplicazione o preprocessing aggiuntivo

### 2. Student Model Predictor (`predict_student.py`)
- Carica il modello addestrato (`Improved_Phi_GRU_ATT`)
- Preprocessing audio identico al training
- Progress bar durante le predizioni

### 3. BirdNET Predictor (`predict_birdnet.py`)
- Filtra automaticamente per le specie del progetto
- Usa file temporaneo per species filtering
- Cleanup automatico delle risorse

### 4. Model Comparator (`compare_predictions.py`)
- Metriche complete: accuracy, precision, recall, F1-score
- Matrici di confusione per entrambi i modelli
- Analisi di accordo/disaccordo tra modelli
- Accuratezza per classe
- Report dettagliato in formato testo

## 📈 Metriche Calcolate

- **Overall Accuracy**: Accuratezza complessiva
- **Per-Class Metrics**: Precision, recall, F1 per ogni specie
- **Confusion Matrices**: Visualizzazione errori di classificazione
- **Agreement Analysis**: 
  - Entrambi corretti
  - Solo student corretto
  - Solo BirdNET corretto
  - Entrambi sbagliati
- **Confidence Analysis**: Distribuzione delle confidenze

## 🔧 Personalizzazione

### 🐦 Aggiungere Nuove Specie

Per aggiungere nuove specie di uccelli al sistema, segui questi passaggi:

#### 1. **Aggiorna la Configurazione Principale**
Modifica `config/bird_classification.yaml`:

```yaml
dataset:
  allowed_bird_classes:
    - "Bubo_bubo"
    - "Certhia_familiaris" 
    - "Apus_apus"
    - "Certhia_brachydactyla"
    - "Emberiza_cia"
    - "Lophophanes_cristatus"
    - "Periparus_ater"
    - "Poecile_montanus"
    - "TUA_NUOVA_SPECIE"  # ⬅️ Aggiungi qui (formato: Genus_species)
    # "non-bird" viene aggiunta automaticamente

model:
  num_classes: 10  # ⬅️ Aggiorna il numero (8 specie + no_birds + nuova specie)
```

#### 2. **Aggiorna il Mapping BirdNET**
Modifica `benchmark/predict_birdnet.py` nella funzione `_create_species_list()`:

```python
species_mapping = {
    'Bubo_bubo': 'Bubo bubo_Eurasian Eagle-Owl',
    'Apus_apus': 'Apus apus_Common Swift', 
    'Certhia_familiaris': 'Certhia familiaris_Eurasian Treecreeper',
    'Certhia_brachydactyla': 'Certhia brachydactyla_Short-toed Treecreeper',
    'Emberiza_cia': 'Emberiza cia_Rock Bunting',
    'Lophophanes_cristatus': 'Lophophanes cristatus_Crested Tit',
    'Periparus_ater': 'Periparus ater_Coal Tit',
    'Poecile_montanus': 'Poecile montanus_Willow Tit',
    'TUA_NUOVA_SPECIE': 'Genus species_Common Name',  # ⬅️ Aggiungi mapping
    'no_birds': 'no_birds'
}
```

#### 3. **Verifica Supporto BirdNET**
BirdNET supporta **oltre 3000 specie** globalmente. Per verificare se la tua specie è supportata:

- **Nome scientifico**: Usa il formato `Genus species` (es. `Turdus merula`)
- **Nome comune inglese**: Spesso richiesto (es. `Common Blackbird`)
- **Database BirdNET**: Include specie europee, nordamericane e molte altre

**Specie europee comuni supportate:**
- `Turdus merula_Common Blackbird`
- `Parus major_Great Tit`
- `Sylvia atricapilla_Blackcap`
- `Fringilla coelebs_Common Chaffinch`
- `Erithacus rubecula_European Robin`

#### 4. **Organizza i Dati Audio**
Crea directory per la nuova specie:

```bash
bird_sound_dataset/
├── TUA_NUOVA_SPECIE/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── ...
```

**Formato richiesto:**
- **Sample rate**: 32kHz (conversione automatica)
- **Durata**: ≥3 secondi (clip più corti vengono paddati)
- **Formato**: WAV, MP3, FLAC supportati
- **Qualità**: Audio puliti con vocalizzi chiari

#### 5. **Ri-addestra il Modello**
```bash
# Dal directory root
python train.py

# Il modello rileverà automaticamente la nuova classe
# e aggiornerà le dimensioni dell'output layer
```

#### 6. **Aggiorna i Benchmark**
Le configurazioni di benchmark rileveranno automaticamente le nuove specie dalle configurazioni, ma verifica che:

- **Tutte le configurazioni YAML** puntino al config aggiornato
- **Test di validazione** includano la nuova specie
- **Soglie di confidenza** siano appropriate per la nuova specie

#### 7. **Considerazioni Importanti**

**🎯 Qualità dei Dati:**
- Minimum **50-100 esempi** per specie per training decente
- **Varietà**: Diversi individui, condizioni di registrazione, stagioni
- **Bilanciamento**: Cerca di avere numeri simili tra le specie

**🔄 Preprocessing Coerente:**
- Il sistema applica automaticamente il preprocessing identico
- **Extract calls**: `true` per rilevare automaticamente i vocalizzi
- **Bandpass filter**: 150-16000 Hz applicato automaticamente

**⚠️ Limitazioni BirdNET:**
- Se la specie **non è supportata** da BirdNET, i benchmark mostreranno sempre `no_birds`
- **Alternative**: Usa benchmark `birds_only` per escludere confronti problematici
- **Testing**: Usa `test_benchmark` per validare rapidamente nuove aggiunte

#### 8. **Test di Validazione**
```bash
# Test rapido con nuova specie
./run_docker_benchmark.sh test_new_species_gpu0 0 test_benchmark

# Verifica che la nuova specie appaia nei risultati
```

**Esempio Completo - Aggiungere Merlo (Turdus merula):**

1. **Config**: `"Turdus_merula"` in `allowed_bird_classes`
2. **BirdNET**: `'Turdus_merula': 'Turdus merula_Common Blackbird'`
3. **Dati**: Directory `bird_sound_dataset/Turdus_merula/`
4. **Training**: `python train.py`
5. **Test**: `./run_docker_benchmark.sh test_merlo_gpu0 0 test_benchmark`

### Modificare Soglie di Confidenza
```yaml
student_model:
  inference:
    confidence_threshold: 0.05  # Soglia bassa per catturare più predizioni

birdnet:
  confidence_threshold: 0.25  # Soglia base per rilevamenti
  no_birds_threshold: 0.4     # Soglia separata per no_birds (più alta = meno no_birds)
  use_adaptive_threshold: true  # Abilita soglie adattive
  adaptive_factor: 1.6         # Moltiplicatore per decisioni no_birds
```

### Configurare Modalità Birds-Only
```yaml
benchmark:
  mode:
    birds_only: true  # Escludi no_birds dalle metriche
    exclude_no_birds_from_ground_truth: true  # Rimuovi campioni no_birds dal test set
    force_fair_preprocessing: true  # Usa preprocessing identico per entrambi i modelli
```

### Cambiare Output Format
```yaml
comparison:
  save_plots: true
  save_detailed_json: true
  save_confusion_matrices: true
```

## 🐛 Troubleshooting

### Errori Comuni

**"No audio files found"**
- Verifica che `bird_sound_dataset/` esista
- Controlla i formati supportati (.wav, .mp3, .flac)

**"Model loading failed"**
- Verifica il path del modello in `quick_start.yaml`
- Controlla compatibilità architettura modello

**"BirdNET species not found"**
- Alcune specie potrebbero non essere in BirdNET
- Controlla la lista supportata online

### Performance Tips

**🚀 Parametri di Debug Supportati:**
```bash
# Limitare numero di file totali
./run_docker_benchmark.sh test 0 benchmark debug.files_limit=20

# Limitare file per classe
./run_docker_benchmark.sh test 0 benchmark debug.max_files_per_class=5

# Combinare entrambi
./run_docker_benchmark.sh test 0 benchmark debug.files_limit=50 debug.max_files_per_class=10
```

**🔧 Ottimizzazione Velocità:**
- **Quick test**: `debug.files_limit=5` per test rapidissimi (30 secondi)
- **Development**: `debug.files_limit=20` per sviluppo (2-3 minuti)
- **Pre-production**: `debug.files_limit=100` per test approfonditi
- **Production**: Nessun limite per benchmark completi

**⚡ Caching e Prestazioni:**
- I risultati vengono cachati per evitare ricalcoli
- BirdNET è più lento: considera subset per iterazioni veloci
- GPU accelera solo lo student model (BirdNET è CPU-bound)

## 📝 Log e Debug

I log dettagliati vengono salvati automaticamente da Hydra:
```
benchmark_results/hydra_outputs/[timestamp]/
├── main.log           # Log completo
├── .hydra/
│   ├── config.yaml    # Configurazione usata
│   └── overrides.yaml # Override applicati
```

## 🤝 Contribuire

Per migliorare il sistema:
1. Aggiungi nuove metriche in `compare_predictions.py`
2. Implementa nuovi visualizzazioni
3. Ottimizza preprocessing per velocità
4. Aggiungi supporto per nuovi modelli di riferimento 