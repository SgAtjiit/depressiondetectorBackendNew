"""
Utility functions for the depression detection project
"""
import wave
import contextlib
import numpy as np
import pandas as pd
from typing import List


def get_audio_duration(file_path: str) -> float:
    """
    Get duration of audio file in seconds
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Duration in seconds
    """
    try:
        with contextlib.closing(wave.open(file_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def get_dataset_statistics(participant_ids: List[str], dataset_path: str):
    """
    Calculate statistics about audio durations in dataset
    
    Args:
        participant_ids: List of participant IDs
        dataset_path: Path to dataset directory
    """
    durations = []
    
    for pid in participant_ids:
        file_path = f"{dataset_path}/{pid}_P/{pid}_P/{pid}_AUDIO.wav"
        duration = get_audio_duration(file_path)
        if duration is not None:
            durations.append(duration)
    
    if durations:
        print(f"Dataset Audio Statistics:")
        print(f"  Total files: {len(durations)}")
        print(f"  Average duration: {np.mean(durations):.1f}s")
        print(f"  Min duration: {np.min(durations):.1f}s")
        print(f"  Max duration: {np.max(durations):.1f}s")
        print(f"  Median duration: {np.median(durations):.1f}s")
    else:
        print("No valid audio files found")


def check_class_balance(y: pd.Series, split_name: str = ""):
    """
    Check and print class balance
    
    Args:
        y: Series of labels
        split_name: Name of the split (e.g., "Train", "Test")
    """
    counts = y.value_counts()
    total = len(y)
    
    print(f"\n{split_name} Class Distribution:")
    print(f"  Not Depressed (0): {counts.get(0, 0)} ({counts.get(0, 0)/total*100:.1f}%)")
    print(f"  Depressed (1):     {counts.get(1, 0)} ({counts.get(1, 0)/total*100:.1f}%)")
    print(f"  Total:             {total}")
    
    # Calculate imbalance ratio
    if len(counts) == 2:
        imbalance_ratio = counts.max() / counts.min()
        print(f"  Imbalance ratio:   {imbalance_ratio:.2f}:1")


def save_predictions_to_csv(predictions, labels, output_file):
    """
    Save predictions to CSV file
    
    Args:
        predictions: Array of predictions
        labels: Array of true labels
        output_file: Path to output CSV file
    """
    df = pd.DataFrame({
        'true_label': labels,
        'predicted_label': predictions,
        'correct': labels == predictions
    })
    
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to: {output_file}")
    
    # Print summary
    accuracy = (labels == predictions).mean()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Correct predictions: {(labels == predictions).sum()}/{len(labels)}")


def create_summary_report(results_dict: dict, output_file: str = None):
    """
    Create a summary report of model results
    
    Args:
        results_dict: Dictionary with model results
        output_file: Optional path to save report
    """
    report = []
    report.append("="*60)
    report.append("DEPRESSION DETECTION MODEL - SUMMARY REPORT")
    report.append("="*60)
    
    for model_name, metrics in results_dict.items():
        report.append(f"\n{model_name}")
        report.append("-"*60)
        report.append(f"Accuracy: {metrics['accuracy']:.4f}")
        
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            report.append(f"Confusion Matrix:")
            report.append(f"  True Negatives:  {cm[0,0]}")
            report.append(f"  False Positives: {cm[0,1]}")
            report.append(f"  False Negatives: {cm[1,0]}")
            report.append(f"  True Positives:  {cm[1,1]}")
    
    report_text = "\n".join(report)
    print(report_text)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {output_file}")
    
    return report_text