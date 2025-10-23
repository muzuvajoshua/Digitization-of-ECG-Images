"""
Visualization utilities for ECG data and images
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


def plot_ecg_signal(signal: np.ndarray,
                    fs: float,
                    lead_name: str = "",
                    title: str = "",
                    figsize: Tuple[int, int] = (15, 4),
                    ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot a single ECG signal.

    Args:
        signal: ECG signal array
        fs: Sampling frequency in Hz
        lead_name: Name of the lead
        title: Plot title
        figsize: Figure size
        ax: Matplotlib axes object (if None, creates new figure)

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Create time axis
    time = np.arange(len(signal)) / fs

    # Plot signal
    ax.plot(time, signal, 'b-', linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (mV)')

    # Set title
    if title:
        ax.set_title(title)
    elif lead_name:
        ax.set_title(f'ECG Lead {lead_name}')

    return ax


def plot_all_leads(leads_data: Dict[str, np.ndarray],
                   fs: float,
                   title: str = "12-Lead ECG",
                   figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
    """
    Plot all 12 ECG leads in a grid layout.

    Args:
        leads_data: Dictionary mapping lead names to signal arrays
        fs: Sampling frequency in Hz
        title: Overall plot title
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    fig, axes = plt.subplots(6, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, lead_name in enumerate(lead_names):
        if lead_name in leads_data:
            plot_ecg_signal(
                leads_data[lead_name],
                fs,
                lead_name=lead_name,
                ax=axes[idx]
            )
        else:
            axes[idx].text(0.5, 0.5, f'{lead_name}\n(No data)',
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])

    fig.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout()

    return fig


def plot_ecg_comparison(predicted: np.ndarray,
                       ground_truth: np.ndarray,
                       fs: float,
                       lead_name: str = "",
                       figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Plot predicted vs ground truth ECG signals for comparison.

    Args:
        predicted: Predicted ECG signal
        ground_truth: Ground truth ECG signal
        fs: Sampling frequency
        lead_name: Name of the lead
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot ground truth
    time_gt = np.arange(len(ground_truth)) / fs
    axes[0].plot(time_gt, ground_truth, 'g-', linewidth=0.8, label='Ground Truth')
    axes[0].set_ylabel('Amplitude (mV)')
    axes[0].set_title(f'Ground Truth - Lead {lead_name}')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot predicted
    time_pred = np.arange(len(predicted)) / fs
    axes[1].plot(time_pred, predicted, 'b-', linewidth=0.8, label='Predicted')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude (mV)')
    axes[1].set_title(f'Predicted - Lead {lead_name}')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()

    return fig


def plot_ecg_overlay(predicted: np.ndarray,
                     ground_truth: np.ndarray,
                     fs: float,
                     lead_name: str = "",
                     figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Overlay predicted and ground truth ECG signals.

    Args:
        predicted: Predicted ECG signal
        ground_truth: Ground truth ECG signal
        fs: Sampling frequency
        lead_name: Name of the lead
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Ensure equal length for overlay
    min_length = min(len(predicted), len(ground_truth))
    predicted = predicted[:min_length]
    ground_truth = ground_truth[:min_length]

    time = np.arange(min_length) / fs

    # Plot both signals
    ax.plot(time, ground_truth, 'g-', linewidth=1.2, label='Ground Truth', alpha=0.7)
    ax.plot(time, predicted, 'b-', linewidth=1.0, label='Predicted', alpha=0.7)

    # Plot error
    error = np.abs(ground_truth - predicted)
    ax.fill_between(time, 0, error, alpha=0.2, color='red', label='Absolute Error')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_title(f'ECG Comparison - Lead {lead_name}')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    return fig


def display_ecg_image(image_path: Path,
                     title: str = "",
                     figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Display an ECG image.

    Args:
        image_path: Path to the image file
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    # Read image
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_rgb)
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=14)

    plt.tight_layout()

    return fig


def display_multiple_segments(ecg_id: str,
                              segments_dir: Path,
                              segments: List[str] = None,
                              figsize: Tuple[int, int] = (15, 15)) -> plt.Figure:
    """
    Display multiple image segments for the same ECG.

    Args:
        ecg_id: ECG record ID
        segments_dir: Directory containing the segments
        segments: List of segment IDs to display (e.g., ['0001', '0003', '0005'])
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    if segments is None:
        segments = ['0001', '0003', '0004', '0005', '0006', '0009', '0010', '0011', '0012']

    # Calculate grid size
    n_segments = len(segments)
    n_cols = 3
    n_rows = (n_segments + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

    segment_names = {
        '0001': 'Original',
        '0003': 'Color scan',
        '0004': 'BW scan',
        '0005': 'Photo',
        '0006': 'Screen photo',
        '0009': 'Stained',
        '0010': 'Damaged',
        '0011': 'Mold color',
        '0012': 'Mold BW'
    }

    for idx, segment in enumerate(segments):
        image_path = segments_dir / ecg_id / f"{ecg_id}-{segment}.png"

        if image_path.exists():
            image = cv2.imread(str(image_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(image_rgb)
            axes[idx].set_title(f"{segment}: {segment_names.get(segment, 'Unknown')}")
        else:
            axes[idx].text(0.5, 0.5, f'{segment}\n(Not found)',
                          ha='center', va='center', transform=axes[idx].transAxes)

        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(len(segments), len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'ECG ID: {ecg_id} - All Segments', fontsize=16)
    plt.tight_layout()

    return fig


def plot_image_with_signals(image_path: Path,
                           leads_data: Dict[str, np.ndarray],
                           fs: float,
                           figsize: Tuple[int, int] = (18, 12)) -> plt.Figure:
    """
    Display ECG image alongside extracted signals.

    Args:
        image_path: Path to ECG image
        leads_data: Dictionary of extracted signals
        fs: Sampling frequency
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    # Create figure with custom layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=0.3)

    # Top: Image
    ax_img = fig.add_subplot(gs[0])
    image = cv2.imread(str(image_path))
    if image is not None:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax_img.imshow(image_rgb)
    ax_img.axis('off')
    ax_img.set_title('ECG Image', fontsize=14)

    # Bottom: Signals
    ax_signals = fig.add_subplot(gs[1])

    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    offset = 0
    spacing = 2.0  # mV spacing between leads

    for lead_name in lead_names:
        if lead_name in leads_data:
            signal = leads_data[lead_name]
            time = np.arange(len(signal)) / fs
            ax_signals.plot(time, signal + offset, linewidth=0.8, label=lead_name)
            offset += spacing

    ax_signals.set_xlabel('Time (s)')
    ax_signals.set_ylabel('Amplitude (mV)')
    ax_signals.set_title('Extracted ECG Signals', fontsize=14)
    ax_signals.legend(loc='right', fontsize=8)
    ax_signals.grid(True, alpha=0.3)

    return fig


def plot_training_history(history: Dict,
                         figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
    """
    Plot training history (loss, metrics over epochs).

    Args:
        history: Dictionary containing training history
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot loss
    if 'loss' in history:
        axes[0].plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot metric (e.g., SNR)
    if 'snr' in history:
        axes[1].plot(history['snr'], label='Training SNR')
    if 'val_snr' in history:
        axes[1].plot(history['val_snr'], label='Validation SNR')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('SNR (dB)')
    axes[1].set_title('Training and Validation SNR')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


if __name__ == "__main__":
    print("ECG Visualization utilities loaded successfully!")

    # Create sample ECG signal for testing
    fs = 500
    duration = 2.5
    t = np.linspace(0, duration, int(fs * duration))
    sample_signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)

    # Test plotting
    fig = plot_ecg_signal(sample_signal, fs, lead_name="Test", title="Sample ECG Signal")
    print("Sample plot created successfully!")
    plt.close()
