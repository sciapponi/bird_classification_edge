"""
Audio Processing Utilities

This module provides functions for audio processing tasks like filtering,
segmentation, and feature extraction.
"""

import os
import time
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import find_peaks, butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Create a Butterworth bandpass filter.
    
    Args:
        lowcut (float): Low cutoff frequency in Hz
        highcut (float): High cutoff frequency in Hz
        fs (float): Sampling rate in Hz
        order (int): Filter order
        
    Returns:
        tuple: (b, a) filter coefficients
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a Butterworth bandpass filter to the data.
    
    Args:
        data (ndarray): Input audio signal
        lowcut (float): Low cutoff frequency in Hz
        highcut (float): High cutoff frequency in Hz
        fs (float): Sampling rate in Hz
        order (int): Filter order
        
    Returns:
        ndarray: Filtered audio signal
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def compute_adaptive_parameters(y, sr, lowcut, highcut):
    """
    Compute adaptive parameters for bird call detection based on the overall filtered energy.
    
    Args:
        y (ndarray): Audio signal
        sr (float): Sampling rate in Hz
        lowcut (float): Low cutoff frequency in Hz
        highcut (float): High cutoff frequency in Hz
        
    Returns:
        tuple: (adaptive_prominence, adaptive_energy_threshold)
    """
    # Filter the audio to the bird call frequency band
    y_filtered = apply_bandpass_filter(y, lowcut, highcut, sr, order=4)
    
    # Compute a short-time amplitude envelope using a moving average
    frame_length = int(sr * 0.05)  # 50 ms window
    hop_length = int(sr * 0.01)    # 10 ms hop
    envelope_frames = librosa.util.frame(np.abs(y_filtered), frame_length=frame_length, hop_length=hop_length)
    envelope = envelope_frames.mean(axis=0)
    
    # Use the median and MAD (median absolute deviation) as robust measures
    median_env = np.median(envelope)
    mad_env = np.median(np.abs(envelope - median_env))
    
    # Set the adaptive prominence: median + k * MAD (tune k as needed)
    adaptive_prominence = median_env + 1.5 * mad_env

    # Compute overall RMS energy of the filtered signal
    rms_all = np.sqrt(np.mean(y_filtered ** 2))
    
    # Set the adaptive energy threshold (this is the baseline, and will be lowered for background verification)
    adaptive_energy_threshold = 0.5 * rms_all

    return adaptive_prominence, adaptive_energy_threshold

def extract_call_segments(audio_path, output_folder=None, clip_duration=3.0, sr=22050,
                         lowcut=2000, highcut=10000, min_peak_distance=1.0, height_percentile=75,
                         verbose=False, save_clips=False):
    """
    Detects bird calls by applying a bandpass filter and using an adaptive threshold
    for peak detection, then extracts clips centered on detected peaks.
    
    Args:
        audio_path (str): Path to audio file
        output_folder (str, optional): Directory to save extracted clips
        clip_duration (float): Duration of extracted clips in seconds
        sr (int): Sample rate in Hz
        lowcut (int): Low frequency cutoff for bandpass filter in Hz
        highcut (int): High frequency cutoff for bandpass filter in Hz
        min_peak_distance (float): Minimum time between peaks in seconds
        height_percentile (float): Percentile threshold for peak height
        verbose (bool): Whether to print detailed information
        save_clips (bool): Whether to save extracted clips to disk
        
    Returns:
        tuple: (call_intervals, segments, original_audio, sample_rate, duration)
    """
    print(f"Processing file: {audio_path}")
    start_time = time.time()
    
    # Create output folder if needed
    if save_clips and output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    # Check if file exists
    if not os.path.exists(audio_path):
        if verbose:
            print(f"File not found: {audio_path}")
        return [], None, None, None, 0
    
    try:
        # Load audio with downsampling if needed
        print(f"Loading audio with librosa...")
        y, sr = librosa.load(audio_path, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"Audio loaded: {duration:.2f} seconds, sr={sr}")
    
        # Get the filename without extension for naming saved clips
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Compute adaptive parameters for detection
        print("Calculating adaptive parameters...")
        adaptive_prominence, _ = compute_adaptive_parameters(y, sr, lowcut, highcut)
        
        # Filter the full audio for detection
        print(f"Applying bandpass filter {lowcut}-{highcut} Hz...")
        y_filtered = apply_bandpass_filter(y, lowcut, highcut, sr, order=4)
        
        # Compute amplitude envelope from the filtered signal
        print("Computing envelope...")
        frame_length = int(sr * 0.05)
        hop_length = int(sr * 0.01)
        envelope_frames = librosa.util.frame(np.abs(y_filtered), frame_length=frame_length, hop_length=hop_length)
        envelope = envelope_frames.mean(axis=0)
        
        # Calculate minimum distance between peaks in frames
        min_peak_distance_frames = int(min_peak_distance / (hop_length / sr))
        
        # Detect peaks using the adaptive prominence and minimum distance
        print("Detecting peaks...")
        peaks, properties = find_peaks(envelope, 
                                    prominence=adaptive_prominence,
                                    distance=min_peak_distance_frames)
        
        # Handle case with no peaks detected
        if len(peaks) == 0:
            print("No peaks detected!")
            if verbose:
                print(f"No peaks detected in {audio_path}. Try adjusting detection parameters.")
            return [], [], y, sr, duration
        
        print(f"Detected {len(peaks)} raw peaks")
        
        # Sort peaks by prominence (highest first)
        sorted_indices = np.argsort(-properties['prominences'])
        sorted_peaks = peaks[sorted_indices]
        sorted_prominences = properties['prominences'][sorted_indices]
        
        # Keep only the top percentile of peaks based on height/amplitude
        if len(sorted_peaks) > 0:  # Check if any peaks were found
            height_threshold = np.percentile(envelope[sorted_peaks], height_percentile)
            selected_peaks = [p for i, p in enumerate(sorted_peaks) if envelope[p] >= height_threshold]
        else:
            selected_peaks = []
        
        # Convert peaks to time
        peak_times = librosa.frames_to_time(selected_peaks, sr=sr, hop_length=hop_length)
        
        print(f"Selected {len(peak_times)} significant peaks")
        if verbose:
            print(f"Detected {len(peak_times)} significant bird calls in {audio_path}")

        call_intervals = []
        segments = []
        for i, t in enumerate(peak_times):
            start_time_sec = max(0, t - clip_duration / 2)  # Ensure we don't go below 0
            end_time_sec = min(duration, t + clip_duration / 2)  # Ensure we don't exceed audio length

            start_sample = int(start_time_sec * sr)
            end_sample = int(end_time_sec * sr)
            segment = y[start_sample:end_sample]
            segments.append(segment)

            # Save the clip if requested
            if save_clips and output_folder:
                filename = os.path.join(output_folder, f"{base_filename}_call_{i+1:03d}.wav")
                sf.write(filename, segment, sr)
                if verbose:
                    print(f"Saved call clip: {filename}")

            call_intervals.append((start_time_sec, end_time_sec))
        
        processing_time = time.time() - start_time
        print(f"Extracted {len(segments)} audio segments in {processing_time:.2f} seconds")
        return call_intervals, segments, y, sr, duration
    except Exception as e:
        print(f"ERROR processing {audio_path}: {str(e)}")
        if verbose:
            print(f"Error processing {audio_path}: {str(e)}")
        return [], [], None, None, 0

def extract_empty_segments(audio_path, clip_duration=3.0, sr=22050,
                           lowcut=2000, highcut=10000, energy_threshold_factor=0.5,
                           max_segments_per_file=5, verbose=False):
    """
    Detects segments with low energy (likely no bird calls) and extracts them.
    
    Args:
        audio_path (str): Path to audio file
        clip_duration (float): Duration of extracted clips in seconds
        sr (int): Sample rate in Hz
        lowcut (int): Low frequency cutoff for bandpass filter in Hz
        highcut (int): High frequency cutoff for bandpass filter in Hz
        energy_threshold_factor (float): Factor to multiply median envelope for silence threshold
        max_segments_per_file (int): Maximum number of segments to extract per file
        verbose (bool): Whether to print detailed information
        
    Returns:
        list: List of empty intervals as (start_time, end_time) tuples
    """
    if not os.path.exists(audio_path):
        if verbose:
            print(f"WARNING: File not found during empty segment search: {audio_path}")
        return []

    try:
        y, current_sr = librosa.load(audio_path, sr=None) # Load original SR first
        if current_sr != sr:
            y = librosa.resample(y, orig_sr=current_sr, target_sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)

        if duration < clip_duration:
             if verbose:
                  print(f"Skipping {audio_path}, duration {duration:.2f}s shorter than clip duration {clip_duration:.2f}s.")
             return []

        y_filtered = apply_bandpass_filter(y, lowcut, highcut, sr, order=4)
        
        frame_length = int(sr * 0.05)  # 50 ms window for envelope
        hop_length = int(sr * 0.01)    # 10 ms hop
        
        # Ensure frame_length is not zero, can happen with very short clips or sr
        if frame_length == 0:
            if verbose:
                print(f"Skipping {audio_path}, frame_length for envelope is zero (sr: {sr}, duration: {duration}).")
            return []

        envelope_frames = librosa.util.frame(np.abs(y_filtered), frame_length=frame_length, hop_length=hop_length)
        envelope = envelope_frames.mean(axis=0) # Mean of absolute values in each frame

        if len(envelope) == 0:
            if verbose:
                print(f"Skipping {audio_path}, envelope is empty.")
            return []

        median_env = np.median(envelope)
        silence_threshold = median_env * energy_threshold_factor

        # Find segments where envelope is below the silence threshold
        silent_frames = np.where(envelope < silence_threshold)[0]
        
        if len(silent_frames) == 0:
            return []

        # Group consecutive silent frames
        grouped_silent_frames = np.split(silent_frames, np.where(np.diff(silent_frames) != 1)[0] + 1)
        
        empty_intervals = []
        samples_per_clip = int(clip_duration * sr)
        
        for group in grouped_silent_frames:
            if len(group) * hop_length >= samples_per_clip: # Check if group is long enough in samples
                start_frame = group[0]
                end_frame = group[-1]
                
                start_time_sec = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
                end_time_sec = librosa.frames_to_time(end_frame, sr=sr, hop_length=hop_length)

                # Extract multiple non-overlapping clips from this long silent segment
                current_segment_start = start_time_sec
                while current_segment_start + clip_duration <= end_time_sec and len(empty_intervals) < max_segments_per_file:
                    empty_intervals.append((current_segment_start, current_segment_start + clip_duration))
                    current_segment_start += clip_duration # Move to the start of the next potential clip
            if len(empty_intervals) >= max_segments_per_file:
                break 
        
        return empty_intervals

    except Exception as e:
        print(f"ERROR processing for empty segments {audio_path}: {str(e)}")
        return []

# Utility to count total audio files for progress bar in dataset factory
def count_audio_files(directory, allowed_extensions=('.wav', '.mp3', '.ogg', '.flac')):
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(tuple(allowed_extensions)):
                count += 1
    return count 