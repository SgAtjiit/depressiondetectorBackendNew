"""
Data loading and preprocessing functions
"""
import os
import pandas as pd
from typing import Tuple, List


def normalize_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize label column names to 'PHQ8_Binary'
    """
    candidates = ['PHQ8_Binary', 'PHQ_Binary', 'PHQ8_binary', 'PHQ_binary', 'phq8_binary', 'PHQ8']
    for c in candidates:
        if c in df.columns:
            df = df.rename(columns={c: 'PHQ8_Binary'})
            break
    
    # If still not present, try to find something with "PHQ" substring
    if 'PHQ8_Binary' not in df.columns:
        for c in df.columns:
            if 'PHQ' in c.upper():
                df = df.rename(columns={c: 'PHQ8_Binary'})
                break
    return df


def load_labels(train_path: str, dev_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and normalize label files for train, dev, and test splits
    """
    # Check if files exist
    for p in (train_path, dev_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required label file not found: {p}")
    
    # Load CSVs
    train_labels = pd.read_csv(train_path)
    dev_labels = pd.read_csv(dev_path)
    test_labels = pd.read_csv(test_path)
    
    # Normalize label columns
    train_labels = normalize_label_column(train_labels)
    dev_labels = normalize_label_column(dev_labels)
    test_labels = normalize_label_column(test_labels)
    
    # Normalize Participant_ID column
    for df in [train_labels, dev_labels, test_labels]:
        if 'Participant_ID' not in df.columns and 'participant_ID' in df.columns:
            df.rename(columns={'participant_ID': 'Participant_ID'}, inplace=True)
    
    print(f"Train labels: {train_labels.shape}")
    print(f"Dev labels:   {dev_labels.shape}")
    print(f"Test labels:  {test_labels.shape}")
    
    return train_labels, dev_labels, test_labels


def get_participant_ids(train_labels: pd.DataFrame, 
                       dev_labels: pd.DataFrame, 
                       test_labels: pd.DataFrame) -> List[str]:
    """
    Get unique participant IDs across all splits
    """
    train_ids = train_labels['Participant_ID'].astype(str).tolist()
    dev_ids = dev_labels['Participant_ID'].astype(str).tolist()
    test_ids = test_labels['Participant_ID'].astype(str).tolist()
    
    all_participant_ids = sorted(list(set(train_ids + dev_ids + test_ids)))
    print(f"Total unique participants across splits: {len(all_participant_ids)}")
    
    return all_participant_ids


def merge_features_with_labels(features_df: pd.DataFrame,
                               train_labels: pd.DataFrame,
                               dev_labels: pd.DataFrame,
                               test_labels: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merge extracted features with labels for each split
    """
    # Ensure Participant_ID is string type
    features_df['Participant_ID'] = features_df['Participant_ID'].astype(str)
    train_labels['Participant_ID'] = train_labels['Participant_ID'].astype(str)
    dev_labels['Participant_ID'] = dev_labels['Participant_ID'].astype(str)
    test_labels['Participant_ID'] = test_labels['Participant_ID'].astype(str)
    
    # Merge
    train_df = pd.merge(features_df, train_labels, on='Participant_ID', how='inner')
    dev_df = pd.merge(features_df, dev_labels, on='Participant_ID', how='inner')
    test_df = pd.merge(features_df, test_labels, on='Participant_ID', how='inner')
    
    print(f"train_df: {train_df.shape}")
    print(f"dev_df:   {dev_df.shape}")
    print(f"test_df:  {test_df.shape}")
    
    return train_df, dev_df, test_df


def prepare_train_test_data(train_df: pd.DataFrame,
                           dev_df: pd.DataFrame,
                           test_df: pd.DataFrame,
                           feature_cols: List[str]) -> Tuple:
    """
    Prepare X and y for training and testing
    """
    # Ensure the target column exists
    if 'PHQ8_Binary' not in train_df.columns:
        raise KeyError("PHQ8_Binary not found in merged train_df")
    
    # Build X, y
    X_train = train_df[feature_cols].copy()
    y_train = train_df['PHQ8_Binary'].astype(int).copy()
    
    X_dev = dev_df[feature_cols].copy()
    y_dev = dev_df['PHQ8_Binary'].astype(int).copy()
    
    X_test = test_df[feature_cols].copy()
    y_test = test_df['PHQ8_Binary'].astype(int).copy()
    
    print("\nTrain data:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train distribution:\n{y_train.value_counts()}")
    
    print("\nDev data:")
    print(f"X_dev shape: {X_dev.shape}")
    print(f"y_dev distribution:\n{y_dev.value_counts()}")
    
    print("\nTest data:")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test distribution:\n{y_test.value_counts()}")
    
    return X_train, y_train, X_dev, y_dev, X_test, y_test