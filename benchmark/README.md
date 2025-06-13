# 🐦 Bird Classification Benchmark System

Sistema completo per confrontare le prestazioni del tuo modello di classificazione degli uccelli con BirdNET come riferimento.

## 🚀 Quick Start

```bash
# Dal directory root del progetto
cd benchmark
source ../venv/bin/activate
python run_benchmark.py --config-name=quick_start
```

## 🎯 Cosa fa il Sistema

Il benchmark esegue automaticamente questi passi:

1. **📁 Audio Discovery**: Scopre tutti i file audio dal dataset esistente
2. **🤖 Student Model Predictions**: Genera predizioni con il tuo modello addestrato
3. **🦅 BirdNET Predictions**: Genera predizioni con BirdNET come riferimento
4. **📊 Model Comparison**: Confronta le prestazioni con metriche dettagliate

## 📊 Output e Risultati

Tutti i risultati vengono salvati in `benchmark/benchmark_results/`:

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

### Aggiungere Nuove Specie
1. Aggiorna `config/bird_classification.yaml`
2. Assicurati che BirdNET supporti le specie
3. Ri-addestra il modello se necessario

### Modificare Soglie di Confidenza
```yaml
student_model:
  inference:
    confidence_threshold: 0.1

birdnet:
  confidence_threshold: 0.1
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

- Usa `files_limit: 100` in fase di test
- BirdNET è più lento, considera subset per test rapidi
- I risultati vengono cachati per evitare ricalcoli

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