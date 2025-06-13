"""
Bird Classification Benchmark System

A comprehensive benchmarking pipeline to compare custom bird classification 
models against BirdNET-Analyzer using Hydra configuration management.
"""

__version__ = "1.0.0"
__author__ = "Leo M"

from .predict_student import StudentModelPredictor  
from .predict_birdnet import BirdNETPredictor
from .compare_predictions import ModelComparator

__all__ = [
    "StudentModelPredictor", 
    "BirdNETPredictor",
    "ModelComparator"
] 