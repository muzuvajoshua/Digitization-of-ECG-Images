"""
Evaluation Metrics for ECG Digitization Competition

This module implements the modified Signal-to-Noise Ratio (SNR) metric used
in the PhysioNet ECG Digitization Competition.
"""

import numpy as np
from scipy import signal
from typing import Dict, List, Tuple


def align_signals(predicted: np.ndarray,
                   ground_truth: np.ndarray,
                   fs: float,
                   max_shift_seconds: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align predicted signal with ground truth using cross-correlation.

    Args:
        predicted: Predicted ECG signal
        ground_truth: Ground truth ECG signal
        fs: Sampling frequency in Hz
        max_shift_seconds: Maximum allowed time shift in seconds (default 0.2s)

    Returns:
        Tuple of (aligned_predicted, aligned_ground_truth)
    """
    # Calculate maximum shift in samples
    max_shift_samples = int(max_shift_seconds * fs)

    # Compute cross-correlation
    correlation = signal.correlate(ground_truth, predicted, mode='full')

    # Find the lag that maximizes correlation
    lags = signal.correlation_lags(len(ground_truth), len(predicted), mode='full')

    # Limit the search to within max_shift_samples
    valid_indices = np.where(np.abs(lags) <= max_shift_samples)[0]
    valid_correlation = correlation[valid_indices]
    valid_lags = lags[valid_indices]

    # Find optimal lag
    optimal_lag_idx = np.argmax(valid_correlation)
    optimal_lag = valid_lags[optimal_lag_idx]

    # Align signals based on optimal lag
    if optimal_lag > 0:
        # Predicted signal is ahead, shift it back
        aligned_predicted = predicted[optimal_lag:]
        aligned_ground_truth = ground_truth[:len(aligned_predicted)]
    elif optimal_lag < 0:
        # Predicted signal is behind, shift ground truth
        aligned_ground_truth = ground_truth[-optimal_lag:]
        aligned_predicted = predicted[:len(aligned_ground_truth)]
    else:
        # No shift needed
        aligned_predicted = predicted
        aligned_ground_truth = ground_truth

    # Ensure equal length
    min_length = min(len(aligned_predicted), len(aligned_ground_truth))
    aligned_predicted = aligned_predicted[:min_length]
    aligned_ground_truth = aligned_ground_truth[:min_length]

    return aligned_predicted, aligned_ground_truth


def remove_vertical_offset(predicted: np.ndarray,
                           ground_truth: np.ndarray) -> np.ndarray:
    """
    Remove constant vertical offset between signals.

    Args:
        predicted: Predicted ECG signal (will be adjusted)
        ground_truth: Ground truth ECG signal

    Returns:
        Predicted signal with vertical offset removed
    """
    offset = np.mean(ground_truth) - np.mean(predicted)
    return predicted + offset


def calculate_snr(predicted: np.ndarray,
                  ground_truth: np.ndarray,
                  fs: float,
                  align: bool = True) -> float:
    """
    Calculate Signal-to-Noise Ratio in dB.

    The SNR is calculated as:
    SNR(dB) = 10 * log10(signal_power / noise_power)

    where:
    - signal_power = sum of squared ground truth values
    - noise_power = sum of squared reconstruction errors

    Args:
        predicted: Predicted ECG signal
        ground_truth: Ground truth ECG signal
        fs: Sampling frequency in Hz
        align: Whether to align signals before computing SNR (default True)

    Returns:
        SNR value in decibels
    """
    # Align signals if requested
    if align:
        predicted, ground_truth = align_signals(predicted, ground_truth, fs)
        predicted = remove_vertical_offset(predicted, ground_truth)

    # Ensure equal length
    min_length = min(len(predicted), len(ground_truth))
    predicted = predicted[:min_length]
    ground_truth = ground_truth[:min_length]

    # Calculate signal power (power of true signal)
    signal_power = np.sum(ground_truth ** 2)

    # Calculate noise power (power of reconstruction error)
    error = ground_truth - predicted
    noise_power = np.sum(error ** 2)

    # Avoid division by zero
    if noise_power == 0:
        return np.inf

    # Calculate SNR in dB
    snr_db = 10 * np.log10(signal_power / noise_power)

    return snr_db


def calculate_multi_lead_snr(predicted_leads: Dict[str, np.ndarray],
                             ground_truth_leads: Dict[str, np.ndarray],
                             fs: float) -> float:
    """
    Calculate SNR for all 12 ECG leads combined.

    The competition metric sums signal and noise powers across all leads
    before computing a single SNR value.

    Args:
        predicted_leads: Dictionary mapping lead names to predicted signals
        ground_truth_leads: Dictionary mapping lead names to ground truth signals
        fs: Sampling frequency in Hz

    Returns:
        Combined SNR value in decibels
    """
    total_signal_power = 0
    total_noise_power = 0

    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    for lead_name in lead_names:
        if lead_name not in predicted_leads or lead_name not in ground_truth_leads:
            continue

        predicted = predicted_leads[lead_name]
        ground_truth = ground_truth_leads[lead_name]

        # Align and adjust for offset
        aligned_pred, aligned_gt = align_signals(predicted, ground_truth, fs)
        aligned_pred = remove_vertical_offset(aligned_pred, aligned_gt)

        # Ensure equal length
        min_length = min(len(aligned_pred), len(aligned_gt))
        aligned_pred = aligned_pred[:min_length]
        aligned_gt = aligned_gt[:min_length]

        # Accumulate powers
        total_signal_power += np.sum(aligned_gt ** 2)
        error = aligned_gt - aligned_pred
        total_noise_power += np.sum(error ** 2)

    # Calculate combined SNR
    if total_noise_power == 0:
        return np.inf

    snr_db = 10 * np.log10(total_signal_power / total_noise_power)

    return snr_db


def calculate_competition_score(predicted_records: List[Dict],
                                ground_truth_records: List[Dict]) -> float:
    """
    Calculate the final competition score.

    This is the average SNR across all ECG records in the dataset.

    Args:
        predicted_records: List of dictionaries, each containing:
            - 'leads': Dict mapping lead names to predicted signals
            - 'fs': Sampling frequency
        ground_truth_records: List of dictionaries with same structure for ground truth

    Returns:
        Average SNR across all records in decibels
    """
    snr_scores = []

    for pred_record, gt_record in zip(predicted_records, ground_truth_records):
        record_snr = calculate_multi_lead_snr(
            pred_record['leads'],
            gt_record['leads'],
            pred_record['fs']
        )
        snr_scores.append(record_snr)

    # Average SNR across all records
    average_snr = np.mean(snr_scores)

    return average_snr


# Utility functions for evaluation

def evaluate_single_lead(predicted: np.ndarray,
                        ground_truth: np.ndarray,
                        fs: float,
                        lead_name: str = "unknown") -> Dict:
    """
    Evaluate a single lead and return detailed metrics.

    Args:
        predicted: Predicted ECG signal
        ground_truth: Ground truth ECG signal
        fs: Sampling frequency
        lead_name: Name of the lead for reporting

    Returns:
        Dictionary with evaluation metrics
    """
    # Align signals
    aligned_pred, aligned_gt = align_signals(predicted, ground_truth, fs)
    aligned_pred = remove_vertical_offset(aligned_pred, aligned_gt)

    # Calculate metrics
    snr = calculate_snr(aligned_pred, aligned_gt, fs, align=False)

    # Additional metrics
    mse = np.mean((aligned_gt - aligned_pred) ** 2)
    mae = np.mean(np.abs(aligned_gt - aligned_pred))
    correlation = np.corrcoef(aligned_gt, aligned_pred)[0, 1]

    return {
        'lead_name': lead_name,
        'snr_db': snr,
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'signal_length': len(aligned_gt)
    }


if __name__ == "__main__":
    # Example usage
    print("Testing ECG evaluation metrics...")

    # Create synthetic test signals
    fs = 500  # 500 Hz sampling rate
    duration = 2.5  # 2.5 seconds
    t = np.linspace(0, duration, int(fs * duration))

    # Simulated ECG signal (simplified)
    ground_truth = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)

    # Add some noise to create prediction
    noise = np.random.normal(0, 0.1, len(ground_truth))
    predicted = ground_truth + noise

    # Add time shift
    shift_samples = 10
    predicted = np.roll(predicted, shift_samples)

    # Add vertical offset
    predicted += 0.5

    # Calculate SNR
    snr = calculate_snr(predicted, ground_truth, fs)
    print(f"\nSNR: {snr:.2f} dB")

    # Detailed evaluation
    metrics = evaluate_single_lead(predicted, ground_truth, fs, "Test Lead")
    print("\nDetailed metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            print(f"  {key}: {value}")
        elif isinstance(value, (float, np.floating)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
