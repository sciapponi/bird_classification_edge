
✅ Benchmark realistico completato!
📊 Campioni processati: 4449/4449 (100.0%)
⏱️  Tempo TOTALE medio: 0.79s ± 0.66s
📉 Compressione media: 34.7%

🔍 BREAKDOWN TEMPI:
   📁 Caricamento: 0.177s (22.4%)
   🔧 Preprocessing: 0.518s (65.7%)
   📊 Spettrogrammi: 0.084s (10.6%)
   💾 Salvataggio: 0.010s (1.3%)

================================================================================
📊 STATISTICHE GENERALI DATASET
================================================================================
📁 Directory analizzata: ../bird_sound_dataset
📋 Totale file: 4,449
💾 Dimensione totale: 10.43 GB
🎵 File audio: 4,449
📄 Altri file: 0
📏 Dimensione media per file: 2.40 MB
📐 Dimensione mediana: 848.92 KB
🔹 File più piccolo: 20.91 KB
🔸 File più grande: 122.92 MB

================================================================================
🧪 RISULTATI BENCHMARK REALISTICO (train_distillation.py)
================================================================================
📊 Campioni testati: 4449/4449
✅ Tasso di successo: 100.0%

⏱️  TEMPI DI PROCESSING REALISTICI:
   Tempo TOTALE medio: 0.79s per file
   Deviazione: ±0.66s
   Range: 0.54s - 16.31s

🔍 BREAKDOWN PROCESSING:
   📁 Caricamento audio: 0.177s (22.4%)
   🔧 Preprocessing: 0.518s (65.7%)
   📊 Spettrogrammi: 0.084s (10.6%)
   💾 Salvataggio NPZ: 0.010s (1.3%)

📦 COMPRESSIONE REALISTICA:
   Rapporto medio: 34.7% ± 54.6%
   Input campione: 10.43 GB
   Output campione: 587.55 MB

📈 DETTAGLI SPETTROGRAMMI:
   Frame medi per file: 297
   Campioni audio: 96000 (3.0s @ 32kHz)
   Mel bins: 64, Linear bins: 64 (come train_distillation.py)

================================================================================
🎯 STIME REALISTICHE (basate su train_distillation.py)
================================================================================
🎵 File audio da processare: 4,449
⏱️  Tempo stimato preprocessing: 1.3h
   (Include margine sicurezza +30% per overhead sistema)
💾 Dimensione input: 10.43 GB
💾 Dimensione stimata output: 3.61 GB
📉 Compressione: 34.7%

⏰ BREAKDOWN TEMPO TOTALE:
   📁 Caricamento: 13.1m
   🔧 Preprocessing: 38.4m
   📊 Spettrogrammi: 6.2m
   💾 Salvataggio: 44.9s

📡 STIME TRASFERIMENTO:
Velocità     Dataset Originale  Dataset Processato   Risparmio
--------------------------------------------------------------------------------
  5 Mbps     4.7h               1.6h                 3.1h
 10 Mbps     2.4h               49.3m                1.6h
 25 Mbps     56.9m              19.7m                37.2m
 50 Mbps     28.5m              9.9m                 18.6m
100 Mbps     14.2m              4.9m                 9.3m

💡 RACCOMANDAZIONI SPECIFICHE:
   ⚡ Preprocessing lungo (1.3h)
       → Esegui durante la notte o in background
   ✅ Ottima compressione (34.7%)

💽 STORAGE REQUIREMENTS:
   📦 Storage temporaneo necessario: 4.66 GB
   📀 Storage finale (dopo cleanup): 3.61 GB

⏱️  Analisi completa REALISTICA in 58.5m
(venv) leonardomannini@mac dataset_transfer % 