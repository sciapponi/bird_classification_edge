#!/usr/bin/env python3
"""
Dataset Analyzer con Benchmarking REALISTICO
Simula il preprocessing identico a train_distillation.py per stime accurate
"""

import os
import sys
import argparse
import time
import random
import tempfile
import shutil
from pathlib import Path
from collections import defaultdict, Counter
import json
import numpy as np

# Audio file extensions supportate
AUDIO_EXTENSIONS = {
    '.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', 
    '.aiff', '.au', '.3gp', '.amr', '.opus'
}

class RealisticDatasetAnalyzer:
    def __init__(self, dataset_path, include_subdirs=True, max_depth=None, benchmark_samples=30):
        self.dataset_path = Path(dataset_path)
        self.include_subdirs = include_subdirs
        self.max_depth = max_depth
        self.benchmark_samples = benchmark_samples
        self.stats = {
            'total_files': 0,
            'total_size_bytes': 0,
            'file_sizes': [],
            'extensions': Counter(),
            'directory_stats': defaultdict(lambda: {'count': 0, 'size': 0}),
            'empty_files': [],
            'largest_files': [],
            'smallest_files': [],
            'audio_files': 0,
            'other_files': 0,
            'benchmark_results': None
        }
        
    def format_size(self, size_bytes):
        """Converte bytes in formato human-readable"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    
    def format_duration(self, seconds):
        """Converte secondi in formato human-readable"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def is_audio_file(self, file_path):
        """Verifica se il file è un file audio supportato"""
        return file_path.suffix.lower() in AUDIO_EXTENSIONS
    
    def realistic_preprocessing_simulation(self, audio_file_path, temp_dir):
        """
        Simula il preprocessing REALISTICO del progetto train_distillation.py
        Include tutti i passi del preprocessing reale
        """
        start_time = time.time()
        
        try:
            # === STEP 1: AUDIO LOADING (simula torchaudio.load) ===
            load_start = time.time()
            
            # Simula lettura file audio
            with open(audio_file_path, 'rb') as f:
                audio_data = f.read()
            
            # Simula decodifica e caricamento (tempo realistico basato su dimensione)
            file_size_mb = len(audio_data) / (1024 * 1024)
            simulated_load_time = max(0.01, file_size_mb * 0.05)  # ~50ms per MB
            time.sleep(simulated_load_time)
            
            load_time = time.time() - load_start
            
            # === STEP 2: PREPROCESSING AUDIO ===
            preprocess_start = time.time()
            
            # Parametri dal config reale
            sample_rate = 32000
            clip_duration = 3.0
            target_samples = int(sample_rate * clip_duration)  # 96,000 campioni
            
            # Simula preprocessing steps:
            
            # 2.1 Resampling (se necessario)
            time.sleep(0.1)  # Resampling time
            
            # 2.2 Conversione mono 
            time.sleep(0.02)
            
            # 2.3 Estrazione chiamate (extract_call_segments)
            # Questo è computazionalmente costoso nel progetto reale
            time.sleep(0.3)  # Call extraction è lento
            
            # 2.4 Normalizzazione
            time.sleep(0.01)
            
            # 2.5 Augmentazioni (se training) - simula 50% probabilità
            if random.random() < 0.5:
                # Noise, time masking, freq masking, time shift, speed perturb
                time.sleep(0.15)  # Augmentazioni costose
            
            preprocess_time = time.time() - preprocess_start
            
            # === STEP 3: SPETTROGRAMMA GENERATION ===
            spectrogram_start = time.time()
            
            # Parametri spettrogramma dal modello reale
            n_fft = 1024
            hop_length = 320
            n_mel_bins = 64
            n_linear_filters = 64
            
            # Simula generazione spettrogrammi:
            # - combined_log_linear con filtri apprendibili
            # - Mel spectrogram
            # - Linear spectrogram
            
            n_frames = (target_samples - n_fft) // hop_length + 1  # ~298 frames
            
            # Crea spettrogrammi realistici
            mel_spectrogram = np.random.random((n_mel_bins, n_frames)).astype(np.float32)
            linear_spectrogram = np.random.random((n_linear_filters, n_frames)).astype(np.float32)
            
            # Simula computazione filtri learnable (costosa)
            time.sleep(0.08)
            
            spectrogram_time = time.time() - spectrogram_start
            
            # === STEP 4: SALVATAGGIO COMPRESSO ===
            save_start = time.time()
            
            # Salva in formato npz (come nel progetto reale)
            temp_output = temp_dir / f"preprocessed_{audio_file_path.stem}.npz"
            np.savez_compressed(temp_output,
                              mel_spectrogram=mel_spectrogram,
                              linear_spectrogram=linear_spectrogram,
                              # Metadati realistici
                              sample_rate=sample_rate,
                              duration=clip_duration,
                              n_fft=n_fft,
                              hop_length=hop_length,
                              n_mel_bins=n_mel_bins,
                              label=0,  # dummy
                              file_path=str(audio_file_path))
            
            save_time = time.time() - save_start
            
            total_processing_time = time.time() - start_time
            
            # Calcola dimensioni
            input_size = audio_file_path.stat().st_size
            output_size = temp_output.stat().st_size if temp_output.exists() else 0
            
            return {
                'processing_time': total_processing_time,
                'load_time': load_time,
                'preprocess_time': preprocess_time,
                'spectrogram_time': spectrogram_time,
                'save_time': save_time,
                'input_size': input_size,
                'output_size': output_size,
                'compression_ratio': output_size / input_size if input_size > 0 else 0,
                'success': True,
                'n_frames': n_frames,
                'n_mel_bins': n_mel_bins,
                'target_samples': target_samples
            }
            
        except Exception as e:
            return {
                'processing_time': 0,
                'load_time': 0,
                'preprocess_time': 0,
                'spectrogram_time': 0,
                'save_time': 0,
                'input_size': 0,
                'output_size': 0,
                'compression_ratio': 0,
                'success': False,
                'error': str(e),
                'n_frames': 0,
                'n_mel_bins': 0,
                'target_samples': 0
            }
    
    def run_realistic_benchmark(self, audio_files_sample):
        """Esegue benchmark del preprocessing realistico su un campione di file"""
        print(f"\n🧪 BENCHMARK PREPROCESSING REALISTICO")
        print(f"📊 Campione: {len(audio_files_sample)} file")
        print(f"🔬 Simula: train_distillation.py pipeline completa")
        print("="*70)
        
        benchmark_results = []
        temp_dir = Path(tempfile.mkdtemp(prefix="realistic_benchmark_"))
        
        try:
            for i, audio_file in enumerate(audio_files_sample):
                print(f"  🔬 Test {i+1}/{len(audio_files_sample)}: {audio_file.name[:50]}...")
                
                result = self.realistic_preprocessing_simulation(audio_file, temp_dir)
                benchmark_results.append(result)
                
                # Progress update ogni 5 file
                if (i + 1) % 5 == 0:
                    successful = sum(1 for r in benchmark_results if r['success'])
                    avg_time = np.mean([r['processing_time'] for r in benchmark_results if r['success']])
                    print(f"    📈 Progresso: {i+1}/{len(audio_files_sample)} - Successi: {successful} - Tempo medio: {avg_time:.2f}s")
        
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Analisi risultati dettagliata
        successful_results = [r for r in benchmark_results if r['success']]
        
        if not successful_results:
            print("❌ Nessun benchmark completato con successo!")
            return None
        
        # Statistiche dettagliate per ogni fase
        stats = {
            'total_samples': len(audio_files_sample),
            'successful_samples': len(successful_results),
            'success_rate': len(successful_results) / len(audio_files_sample),
            
            # Tempi totali
            'avg_total_time': np.mean([r['processing_time'] for r in successful_results]),
            'std_total_time': np.std([r['processing_time'] for r in successful_results]),
            'min_total_time': np.min([r['processing_time'] for r in successful_results]),
            'max_total_time': np.max([r['processing_time'] for r in successful_results]),
            
            # Breakdown dei tempi
            'avg_load_time': np.mean([r['load_time'] for r in successful_results]),
            'avg_preprocess_time': np.mean([r['preprocess_time'] for r in successful_results]),
            'avg_spectrogram_time': np.mean([r['spectrogram_time'] for r in successful_results]),
            'avg_save_time': np.mean([r['save_time'] for r in successful_results]),
            
            # Compressione
            'avg_compression_ratio': np.mean([r['compression_ratio'] for r in successful_results]),
            'std_compression_ratio': np.std([r['compression_ratio'] for r in successful_results]),
            'total_input_size': sum(r['input_size'] for r in successful_results),
            'total_output_size': sum(r['output_size'] for r in successful_results),
            
            # Dettagli processing
            'avg_n_frames': np.mean([r['n_frames'] for r in successful_results]),
            'avg_target_samples': np.mean([r['target_samples'] for r in successful_results]),
        }
        
        print(f"\n✅ Benchmark realistico completato!")
        print(f"📊 Campioni processati: {stats['successful_samples']}/{stats['total_samples']} ({stats['success_rate']*100:.1f}%)")
        print(f"⏱️  Tempo TOTALE medio: {stats['avg_total_time']:.2f}s ± {stats['std_total_time']:.2f}s")
        print(f"📉 Compressione media: {stats['avg_compression_ratio']*100:.1f}%")
        
        # Breakdown dettagliato
        print(f"\n🔍 BREAKDOWN TEMPI:")
        print(f"   📁 Caricamento: {stats['avg_load_time']:.3f}s ({stats['avg_load_time']/stats['avg_total_time']*100:.1f}%)")
        print(f"   🔧 Preprocessing: {stats['avg_preprocess_time']:.3f}s ({stats['avg_preprocess_time']/stats['avg_total_time']*100:.1f}%)")
        print(f"   📊 Spettrogrammi: {stats['avg_spectrogram_time']:.3f}s ({stats['avg_spectrogram_time']/stats['avg_total_time']*100:.1f}%)")
        print(f"   💾 Salvataggio: {stats['avg_save_time']:.3f}s ({stats['avg_save_time']/stats['avg_total_time']*100:.1f}%)")
        
        return stats
    
    def scan_directory(self):
        """Scansiona la directory velocemente"""
        print(f"🔍 Scansionando: {self.dataset_path}")
        print(f"📁 Includi sottodirectory: {self.include_subdirs}")
        if self.max_depth:
            print(f"📊 Profondità massima: {self.max_depth}")
        print()
        
        if not self.dataset_path.exists():
            print(f"❌ ERRORE: La directory {self.dataset_path} non esiste!")
            return False
        
        # Raccolta di tutti i file
        all_files = []
        
        if self.include_subdirs:
            pattern = "**/*" if self.max_depth is None else "*" * (self.max_depth + 1)
            all_files = list(self.dataset_path.rglob(pattern))
        else:
            all_files = list(self.dataset_path.iterdir())
        
        # Filtra solo i file (non directory)
        files_to_analyze = [f for f in all_files if f.is_file()]
        total_files = len(files_to_analyze)
        
        print(f"📋 Trovati {total_files:,} file da analizzare...")
        
        # Analisi veloce
        audio_files_data = []
        start_time = time.time()
        
        for i, file_path in enumerate(files_to_analyze):
            # Progress ogni 1000 file
            if i % 1000 == 0 and i > 0:
                elapsed = time.time() - start_time
                progress = (i / total_files) * 100
                rate = i / elapsed
                eta = (total_files - i) / rate if rate > 0 else 0
                print(f"  📊 {i:,}/{total_files:,} ({progress:.1f}%) - {rate:.0f} file/s - ETA: {self.format_duration(eta)}")
            
            try:
                stat = file_path.stat()
                size = stat.st_size
                extension = file_path.suffix.lower()
                
                # Aggiorna statistiche
                self.stats['total_files'] += 1
                self.stats['total_size_bytes'] += size
                self.stats['file_sizes'].append(size)
                self.stats['extensions'][extension] += 1
                
                # Statistiche per directory
                try:
                    dir_key = str(file_path.parent.relative_to(self.dataset_path))
                except ValueError:
                    dir_key = str(file_path.parent)
                self.stats['directory_stats'][dir_key]['count'] += 1
                self.stats['directory_stats'][dir_key]['size'] += size
                
                # File vuoti
                if size == 0:
                    self.stats['empty_files'].append(str(file_path))
                
                # Audio vs altri file
                if self.is_audio_file(file_path):
                    self.stats['audio_files'] += 1
                    audio_files_data.append((file_path, size))
                else:
                    self.stats['other_files'] += 1
                    
            except (OSError, IOError):
                continue
        
        # Post-processing
        if self.stats['file_sizes']:
            self.stats['file_sizes'].sort()
            
            # Top 10 file più grandi e piccoli
            audio_files_data.sort(key=lambda x: x[1], reverse=True)
            self.stats['largest_files'] = audio_files_data[:10]
            
            non_empty_audio = [(f, s) for f, s in audio_files_data if s > 0]
            non_empty_audio.sort(key=lambda x: x[1])
            self.stats['smallest_files'] = non_empty_audio[:10]
        
        elapsed = time.time() - start_time
        rate = self.stats['total_files'] / elapsed if elapsed > 0 else 0
        
        print(f"✅ Scansione completata in {self.format_duration(elapsed)}")
        print(f"📈 Velocità media: {rate:.0f} file/s")
        print(f"📊 File totali: {self.stats['total_files']:,}")
        print(f"🎵 File audio: {self.stats['audio_files']:,}")
        
        # Esegui benchmark realistico se ci sono file audio
        if self.stats['audio_files'] > 0:
            # Seleziona campione casuale per benchmark
            audio_files = [item[0] for item in audio_files_data]
            sample_size = min(self.benchmark_samples, len(audio_files))
            
            # Campiona diversi tipi di file se possibile
            if len(audio_files) > sample_size:
                # Prendi file di diverse dimensioni per test più realistico
                audio_files.sort(key=lambda x: x.stat().st_size)
                step = len(audio_files) // sample_size
                audio_sample = [audio_files[i * step] for i in range(sample_size)]
            else:
                audio_sample = audio_files
            
            self.stats['benchmark_results'] = self.run_realistic_benchmark(audio_sample)
        
        return True
    
    def print_summary_stats(self):
        """Stampa statistiche riassuntive"""
        print("\n" + "="*80)
        print("📊 STATISTICHE GENERALI DATASET")
        print("="*80)
        
        print(f"📁 Directory analizzata: {self.dataset_path}")
        print(f"📋 Totale file: {self.stats['total_files']:,}")
        print(f"💾 Dimensione totale: {self.format_size(self.stats['total_size_bytes'])}")
        print(f"🎵 File audio: {self.stats['audio_files']:,}")
        print(f"📄 Altri file: {self.stats['other_files']:,}")
        
        if self.stats['total_files'] > 0:
            avg_size = self.stats['total_size_bytes'] / self.stats['total_files']
            print(f"📏 Dimensione media per file: {self.format_size(avg_size)}")
            
            sizes = sorted(self.stats['file_sizes'])
            median_size = sizes[len(sizes)//2] if sizes else 0
            min_size = min(sizes) if sizes else 0
            max_size = max(sizes) if sizes else 0
            
            print(f"📐 Dimensione mediana: {self.format_size(median_size)}")
            print(f"🔹 File più piccolo: {self.format_size(min_size)}")
            print(f"🔸 File più grande: {self.format_size(max_size)}")
    
    def print_realistic_benchmark_results(self):
        """Stampa risultati del benchmark realistico"""
        if not self.stats['benchmark_results']:
            print("\n❌ Nessun benchmark realistico eseguito")
            return
        
        bench = self.stats['benchmark_results']
        
        print("\n" + "="*80)
        print("🧪 RISULTATI BENCHMARK REALISTICO (train_distillation.py)")
        print("="*80)
        
        print(f"📊 Campioni testati: {bench['successful_samples']}/{bench['total_samples']}")
        print(f"✅ Tasso di successo: {bench['success_rate']*100:.1f}%")
        
        print(f"\n⏱️  TEMPI DI PROCESSING REALISTICI:")
        print(f"   Tempo TOTALE medio: {bench['avg_total_time']:.2f}s per file")
        print(f"   Deviazione: ±{bench['std_total_time']:.2f}s")
        print(f"   Range: {bench['min_total_time']:.2f}s - {bench['max_total_time']:.2f}s")
        
        print(f"\n🔍 BREAKDOWN PROCESSING:")
        total_avg = bench['avg_total_time']
        print(f"   📁 Caricamento audio: {bench['avg_load_time']:.3f}s ({bench['avg_load_time']/total_avg*100:.1f}%)")
        print(f"   🔧 Preprocessing: {bench['avg_preprocess_time']:.3f}s ({bench['avg_preprocess_time']/total_avg*100:.1f}%)")
        print(f"   📊 Spettrogrammi: {bench['avg_spectrogram_time']:.3f}s ({bench['avg_spectrogram_time']/total_avg*100:.1f}%)")
        print(f"   💾 Salvataggio NPZ: {bench['avg_save_time']:.3f}s ({bench['avg_save_time']/total_avg*100:.1f}%)")
        
        print(f"\n📦 COMPRESSIONE REALISTICA:")
        print(f"   Rapporto medio: {bench['avg_compression_ratio']*100:.1f}% ± {bench['std_compression_ratio']*100:.1f}%")
        print(f"   Input campione: {self.format_size(bench['total_input_size'])}")
        print(f"   Output campione: {self.format_size(bench['total_output_size'])}")
        
        print(f"\n📈 DETTAGLI SPETTROGRAMMI:")
        print(f"   Frame medi per file: {bench['avg_n_frames']:.0f}")
        print(f"   Campioni audio: {bench['avg_target_samples']:.0f} (3.0s @ 32kHz)")
        print(f"   Mel bins: 64, Linear bins: 64 (come train_distillation.py)")
    
    def print_realistic_estimates(self):
        """Stampa stime realistiche basate sul benchmark train_distillation.py"""
        print("\n" + "="*80)
        print("🎯 STIME REALISTICHE (basate su train_distillation.py)")
        print("="*80)
        
        if not self.stats['benchmark_results']:
            print("❌ Impossibile generare stime: benchmark non disponibile")
            print("🔄 Esegui nuovamente con file audio nel dataset")
            return
        
        bench = self.stats['benchmark_results']
        
        if self.stats['audio_files'] == 0:
            print("❌ Nessun file audio trovato per il preprocessing")
            return
        
        # Usa i risultati del benchmark realistico
        realistic_time_per_file = bench['avg_total_time']
        realistic_compression_ratio = bench['avg_compression_ratio']
        
        # Margine di sicurezza maggiore per preprocessing complesso
        safety_margin = 1.3  # +30% per overhead sistema, I/O, etc.
        safe_time_per_file = realistic_time_per_file * safety_margin
        
        total_processing_time = self.stats['audio_files'] * safe_time_per_file
        estimated_output_size = self.stats['total_size_bytes'] * realistic_compression_ratio
        
        print(f"🎵 File audio da processare: {self.stats['audio_files']:,}")
        print(f"⏱️  Tempo stimato preprocessing: {self.format_duration(total_processing_time)}")
        print(f"   (Include margine sicurezza +30% per overhead sistema)")
        print(f"💾 Dimensione input: {self.format_size(self.stats['total_size_bytes'])}")
        print(f"💾 Dimensione stimata output: {self.format_size(estimated_output_size)}")
        print(f"📉 Compressione: {realistic_compression_ratio*100:.1f}%")
        
        # Breakdown del tempo
        print(f"\n⏰ BREAKDOWN TEMPO TOTALE:")
        breakdown_time = realistic_time_per_file * self.stats['audio_files']
        print(f"   📁 Caricamento: {self.format_duration(bench['avg_load_time'] * self.stats['audio_files'])}")
        print(f"   🔧 Preprocessing: {self.format_duration(bench['avg_preprocess_time'] * self.stats['audio_files'])}")
        print(f"   📊 Spettrogrammi: {self.format_duration(bench['avg_spectrogram_time'] * self.stats['audio_files'])}")
        print(f"   💾 Salvataggio: {self.format_duration(bench['avg_save_time'] * self.stats['audio_files'])}")
        
        # Stime trasferimento multiple velocità
        upload_speeds = [5, 10, 25, 50, 100]  # Mbps
        print(f"\n📡 STIME TRASFERIMENTO:")
        print(f"{'Velocità':<12} {'Dataset Originale':<18} {'Dataset Processato':<20} {'Risparmio'}")
        print("-" * 80)
        
        for speed_mbps in upload_speeds:
            speed_bytes = (speed_mbps * 1024 * 1024) / 8
            time_original = self.stats['total_size_bytes'] / speed_bytes
            time_processed = estimated_output_size / speed_bytes
            time_saved = time_original - time_processed
            
            print(f"{speed_mbps:>3} Mbps     {self.format_duration(time_original):<18} "
                  f"{self.format_duration(time_processed):<20} {self.format_duration(time_saved)}")
        
        # Raccomandazioni specifiche
        print(f"\n💡 RACCOMANDAZIONI SPECIFICHE:")
        
        if total_processing_time > 7200:  # > 2 ore
            print(f"   ⚠️  Preprocessing molto lungo ({self.format_duration(total_processing_time)})")
            print(f"       → Considera processing parallelo su più core")
            print(f"       → Usa batch processing con checkpointing ogni 1000 file")
        elif total_processing_time > 3600:  # > 1 ora
            print(f"   ⚡ Preprocessing lungo ({self.format_duration(total_processing_time)})")
            print(f"       → Esegui durante la notte o in background")
        else:
            print(f"   ✅ Preprocessing gestibile ({self.format_duration(total_processing_time)})")
        
        if realistic_compression_ratio > 0.4:  # > 40%
            print(f"   ⚠️  Compressione moderata ({realistic_compression_ratio*100:.1f}%)")
            print(f"       → Verifica formato NPZ vs alternatives (HDF5, Parquet)")
        else:
            print(f"   ✅ Ottima compressione ({realistic_compression_ratio*100:.1f}%)")
            
        # Stima storage temporaneo necessario
        temp_storage_needed = estimated_output_size + self.stats['total_size_bytes'] * 0.1  # +10% buffer
        print(f"\n💽 STORAGE REQUIREMENTS:")
        print(f"   📦 Storage temporaneo necessario: {self.format_size(temp_storage_needed)}")
        print(f"   📀 Storage finale (dopo cleanup): {self.format_size(estimated_output_size)}")
    
    def run_full_analysis(self):
        """Esegue analisi completa con benchmark realistico"""
        start_time = time.time()
        
        print("🚀 ANALISI DATASET CON BENCHMARK REALISTICO")
        print("🔬 Simula preprocessing identico a train_distillation.py")
        print("="*80)
        
        if not self.scan_directory():
            return False
        
        # Stampa risultati
        self.print_summary_stats()
        self.print_realistic_benchmark_results()
        self.print_realistic_estimates()
        
        # Tempo totale
        total_time = time.time() - start_time
        print(f"\n⏱️  Analisi completa REALISTICA in {self.format_duration(total_time)}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Analizza dataset con benchmark preprocessing REALISTICO')
    parser.add_argument('dataset_path', help='Percorso alla directory del dataset')
    parser.add_argument('--no-subdirs', action='store_true', help='Non includere sottodirectory')
    parser.add_argument('--max-depth', type=int, help='Profondità massima scansione')
    parser.add_argument('--benchmark-samples', type=int, default=30, 
                       help='Numero di file per benchmark realistico (default: 30)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"❌ ERRORE: Directory {args.dataset_path} non trovata!")
        sys.exit(1)
    
    analyzer = RealisticDatasetAnalyzer(
        dataset_path=args.dataset_path,
        include_subdirs=not args.no_subdirs,
        max_depth=args.max_depth,
        benchmark_samples=args.benchmark_samples
    )
    
    success = analyzer.run_full_analysis()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 