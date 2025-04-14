"""
Bird Sound Classification Datasets

A collection of datasets and utilities for bird sound classification tasks.
"""

# Import core functionality to expose at the package level
from .audio_utils import (
    butter_bandpass, 
    apply_bandpass_filter, 
    compute_adaptive_parameters,
    extract_call_segments,
    extract_empty_segments
)

from .esc50_dataset import (
    ESC50Dataset,
    download_and_extract_esc50
)

from .bird_dataset import BirdSoundDataset

from .empty_segment_dataset import EmptySegmentDataset

from .dataset_factory import (
    create_combined_dataset,
    LabelAdapterDataset
)

__version__ = "0.1.0" 