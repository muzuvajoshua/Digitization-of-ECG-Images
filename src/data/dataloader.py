"""
Data loading utilities for ECG Digitization competition
"""

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    TRAIN_CSV, TEST_CSV, TRAIN_DIR, TEST_DIR,
    ECG_LEADS, SAMPLE_SUBMISSION
)


class ECGDataLoader:
    """Data loader for ECG images and time-series data."""

    def __init__(self, data_dir: Path = None, mode: str = 'train'):
        """
        Initialize ECG data loader.

        Args:
            data_dir: Root directory containing train/test data
            mode: 'train' or 'test'
        """
        self.mode = mode
        self.data_dir = data_dir

        # Load metadata
        if mode == 'train':
            self.metadata = pd.read_csv(TRAIN_CSV)
            self.images_dir = TRAIN_DIR
        else:
            self.metadata = pd.read_csv(TEST_CSV)
            self.images_dir = TEST_DIR

    def get_record_ids(self) -> List[str]:
        """Get list of all record IDs."""
        return self.metadata['id'].unique().tolist()

    def get_metadata(self, record_id: str) -> Dict:
        """
        Get metadata for a specific record.

        Args:
            record_id: ECG record ID

        Returns:
            Dictionary containing metadata
        """
        record_meta = self.metadata[self.metadata['id'] == record_id].iloc[0]
        return record_meta.to_dict()

    def load_time_series(self, record_id: str) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Load time-series data for a training record.

        Args:
            record_id: ECG record ID

        Returns:
            Tuple of (leads_data dict, sampling_frequency)
        """
        if self.mode != 'train':
            raise ValueError("Time series data only available for training mode")

        # Path to CSV file
        csv_path = self.images_dir / record_id / f"{record_id}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Time series file not found: {csv_path}")

        # Read CSV
        df = pd.read_csv(csv_path)

        # Extract leads
        leads_data = {}
        for lead in ECG_LEADS:
            if lead in df.columns:
                leads_data[lead] = df[lead].values

        # Get sampling frequency from metadata
        metadata = self.get_metadata(record_id)
        fs = metadata['fs']

        return leads_data, fs

    def load_image(self, record_id: str, segment: str = None) -> np.ndarray:
        """
        Load ECG image.

        Args:
            record_id: ECG record ID
            segment: Segment ID (e.g., '0001', '0003'). Only for training mode.
                    If None, loads the test image.

        Returns:
            Image as numpy array (BGR format)
        """
        if self.mode == 'train':
            if segment is None:
                # Load first available segment
                segment = '0001'
            image_path = self.images_dir / record_id / f"{record_id}-{segment}.png"
        else:
            # Test mode - single image per record
            image_path = self.images_dir / f"{record_id}.png"

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        return image

    def get_available_segments(self, record_id: str) -> List[str]:
        """
        Get list of available image segments for a training record.

        Args:
            record_id: ECG record ID

        Returns:
            List of available segment IDs
        """
        if self.mode != 'train':
            return []

        record_dir = self.images_dir / record_id

        if not record_dir.exists():
            return []

        # Find all PNG files
        segment_files = list(record_dir.glob(f"{record_id}-*.png"))
        segments = [f.stem.split('-')[-1] for f in segment_files]

        return sorted(segments)

    def load_record(self, record_id: str, segment: str = None) -> Dict:
        """
        Load complete record (image + time series if training).

        Args:
            record_id: ECG record ID
            segment: Image segment ID (training only)

        Returns:
            Dictionary containing all record data
        """
        metadata = self.get_metadata(record_id)
        image = self.load_image(record_id, segment)

        record = {
            'id': record_id,
            'metadata': metadata,
            'image': image,
            'segment': segment
        }

        # Load time series for training data
        if self.mode == 'train':
            leads_data, fs = self.load_time_series(record_id)
            record['leads'] = leads_data
            record['fs'] = fs
            record['available_segments'] = self.get_available_segments(record_id)

        return record

    def __len__(self) -> int:
        """Return number of records."""
        return len(self.get_record_ids())

    def __iter__(self):
        """Iterate over all records."""
        for record_id in self.get_record_ids():
            yield self.load_record(record_id)


class ECGDataset:
    """
    PyTorch-style dataset for ECG images.
    Can be used with both PyTorch and TensorFlow.
    """

    def __init__(self,
                 record_ids: List[str],
                 data_loader: ECGDataLoader,
                 segment: str = '0001',
                 transform=None):
        """
        Initialize dataset.

        Args:
            record_ids: List of record IDs to include
            data_loader: ECGDataLoader instance
            segment: Which image segment to use (training only)
            transform: Optional image transformation function
        """
        self.record_ids = record_ids
        self.data_loader = data_loader
        self.segment = segment
        self.transform = transform

    def __len__(self) -> int:
        return len(self.record_ids)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single item.

        Args:
            idx: Index

        Returns:
            Dictionary containing image and (if training) target signals
        """
        record_id = self.record_ids[idx]
        record = self.data_loader.load_record(record_id, self.segment)

        # Apply transforms if provided
        if self.transform is not None:
            record['image'] = self.transform(record['image'])

        return record


def load_sample_submission() -> pd.DataFrame:
    """
    Load sample submission file.

    Returns:
        Sample submission dataframe
    """
    return pd.read_parquet(SAMPLE_SUBMISSION)


def create_submission(predictions: Dict[str, Dict[str, np.ndarray]],
                     test_metadata: pd.DataFrame,
                     output_path: Path) -> pd.DataFrame:
    """
    Create submission file from predictions.

    Args:
        predictions: Dictionary mapping record_id to lead predictions
                    Format: {record_id: {lead_name: signal_array}}
        test_metadata: Test metadata dataframe
        output_path: Path to save submission file

    Returns:
        Submission dataframe
    """
    submission_rows = []

    for record_id, leads in predictions.items():
        # Get metadata for this record
        record_meta = test_metadata[test_metadata['id'] == record_id].iloc[0]
        fs = record_meta['fs']
        expected_rows = record_meta['number_of_rows']

        for lead_name, signal in leads.items():
            # Ensure correct length
            if len(signal) < expected_rows:
                # Pad if too short
                signal = np.pad(signal, (0, expected_rows - len(signal)), mode='edge')
            elif len(signal) > expected_rows:
                # Truncate if too long
                signal = signal[:expected_rows]

            # Create rows for this lead
            for row_idx, value in enumerate(signal):
                submission_id = f"{record_id}_{row_idx}_{lead_name}"
                submission_rows.append({
                    'id': submission_id,
                    'value': float(value)
                })

    # Create dataframe
    submission_df = pd.DataFrame(submission_rows)

    # Save to parquet
    submission_df.to_parquet(output_path, index=False)
    print(f"Submission saved to: {output_path}")

    return submission_df


def get_data_statistics(data_loader: ECGDataLoader) -> Dict:
    """
    Calculate statistics about the dataset.

    Args:
        data_loader: ECGDataLoader instance

    Returns:
        Dictionary containing dataset statistics
    """
    record_ids = data_loader.get_record_ids()
    n_records = len(record_ids)

    stats = {
        'n_records': n_records,
        'sampling_frequencies': [],
        'signal_lengths': {},
        'image_sizes': []
    }

    print(f"Analyzing {n_records} records...")

    for i, record_id in enumerate(record_ids[:100]):  # Sample first 100 for speed
        try:
            # Get metadata
            metadata = data_loader.get_metadata(record_id)
            stats['sampling_frequencies'].append(metadata['fs'])

            # Load image to get size
            if data_loader.mode == 'train':
                image = data_loader.load_image(record_id, '0001')
            else:
                image = data_loader.load_image(record_id)

            stats['image_sizes'].append(image.shape[:2])  # (height, width)

            # Get signal lengths for training data
            if data_loader.mode == 'train':
                leads, fs = data_loader.load_time_series(record_id)
                for lead_name, signal in leads.items():
                    if lead_name not in stats['signal_lengths']:
                        stats['signal_lengths'][lead_name] = []
                    stats['signal_lengths'][lead_name].append(len(signal))

        except Exception as e:
            print(f"Error processing {record_id}: {e}")
            continue

    # Calculate summary statistics
    stats['fs_unique'] = np.unique(stats['sampling_frequencies']).tolist()
    stats['image_sizes_unique'] = list(set(map(tuple, stats['image_sizes'])))

    return stats


if __name__ == "__main__":
    print("Testing ECG Data Loader...")

    # Test training data loader
    try:
        train_loader = ECGDataLoader(mode='train')
        print(f"\nTraining records: {len(train_loader)}")

        # Load first record
        record_ids = train_loader.get_record_ids()
        if record_ids:
            first_id = record_ids[0]
            print(f"Loading record: {first_id}")

            record = train_loader.load_record(first_id)
            print(f"  Image shape: {record['image'].shape}")
            print(f"  Sampling frequency: {record['fs']} Hz")
            print(f"  Available segments: {record['available_segments']}")
            print(f"  Leads: {list(record['leads'].keys())}")

    except Exception as e:
        print(f"Error testing training loader: {e}")

    # Test test data loader
    try:
        test_loader = ECGDataLoader(mode='test')
        print(f"\nTest records: {len(test_loader)}")
    except Exception as e:
        print(f"Error testing test loader: {e}")
