import librosa
import numpy as np
import pandas as pd
import os
import soundfile as sf
from tqdm import tqdm
from config import (
    DATASET_DIR,
    LABELS_DIR,
    TRAIN_LABELS_FILE,
    DEV_LABELS_FILE,
    TEST_LABELS_FILE,
    FEATURES_DIR,
    SAMPLE_RATE,
    MAX_DURATION
)

def normalize_label_column(df):
    """Normalize label column names across different CSV formats"""
    candidates = ['PHQ8_Binary', 'PHQ_Binary', 'PHQ8_binary', 'PHQ_binary', 'phq8_binary', 'PHQ8']
    for c in candidates:
        if c in df.columns:
            df = df.rename(columns={c: 'PHQ8_Binary'})
            return df
    for c in df.columns:
        if 'PHQ' in c.upper():
            df = df.rename(columns={c: 'PHQ8_Binary'})
            return df
    raise ValueError(f"Could not find PHQ label column in: {df.columns.tolist()}")

def normalize_participant_id_column(df):
    """Normalize participant ID column names"""
    if 'Participant_ID' in df.columns:
        return df
    elif 'participant_ID' in df.columns:
        df = df.rename(columns={'participant_ID': 'Participant_ID'})
        return df
    elif 'participant_id' in df.columns:
        df = df.rename(columns={'participant_id': 'Participant_ID'})
        return df
    else:
        raise ValueError(f"Could not find Participant_ID column in: {df.columns.tolist()}")

def load_all_labels():
    """Load and combine all label split files"""
    print(f"📂 Reading labels from: {LABELS_DIR}")
    
    for path in [TRAIN_LABELS_FILE, DEV_LABELS_FILE, TEST_LABELS_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required label file not found: {path}")
    
    train_labels = pd.read_csv(TRAIN_LABELS_FILE)
    dev_labels = pd.read_csv(DEV_LABELS_FILE)
    test_labels = pd.read_csv(TEST_LABELS_FILE)
    
    print(f"✅ Train labels: {train_labels.shape}")
    print(f"✅ Dev labels:   {dev_labels.shape}")
    print(f"✅ Test labels:  {test_labels.shape}")
    
    train_labels = normalize_label_column(train_labels)
    train_labels = normalize_participant_id_column(train_labels)
    dev_labels = normalize_label_column(dev_labels)
    dev_labels = normalize_participant_id_column(dev_labels)
    test_labels = normalize_label_column(test_labels)
    test_labels = normalize_participant_id_column(test_labels)
    
    all_labels = pd.concat([train_labels, dev_labels, test_labels], ignore_index=True)
    all_labels['Participant_ID'] = all_labels['Participant_ID'].astype(str)
    
    print(f"\n✅ Combined labels: {all_labels.shape}")
    print(f"   Unique participants: {all_labels['Participant_ID'].nunique()}")
    print(f"   Label distribution:\n{all_labels['PHQ8_Binary'].value_counts()}")
    
    return all_labels

def aggregate_stats(arr):
    """Return mean, std, min, max for a numpy array (flattened)."""
    arr_flat = np.asarray(arr).flatten()
    if arr_flat.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    return (np.mean(arr_flat), np.std(arr_flat), np.min(arr_flat), np.max(arr_flat))

def extract_features_from_file(file_path, sr_target=16000, max_duration=180):
    """
    Extract enhanced features with prosody (depression-specific)
    ✅ SMART DURATION HANDLING:
    - If audio > 180s: Extract first 180s
    - If audio <= 180s: Use full audio
    
    Args:
        file_path: Path to audio file
        sr_target: Target sample rate (default 16000 Hz)
        max_duration: Maximum duration in seconds (default 180s = 3 minutes)
    """
    try:
        audio_info = sf.info(file_path)
        audio_duration = audio_info.duration
    except Exception as e:
        print(f"⚠️  Using librosa for {file_path}: {e}")
        audio_duration = librosa.get_duration(path=file_path)
    
    # ✅ Smart duration selection
    if audio_duration > max_duration:
        duration_to_load = max_duration
        print(f"⏱️  Audio is {audio_duration:.1f}s, extracting first {max_duration}s")
    else:
        duration_to_load = None
    
    # Load audio with selected duration
    try:
        y, sr = librosa.load(file_path, sr=sr_target, duration=duration_to_load)
    except Exception as e:
        print(f"⚠️  Librosa load failed, trying soundfile: {e}")
        # Fallback to soundfile
        y, sr = sf.read(file_path, dtype='float32')
        if sr != sr_target:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
            sr = sr_target
        if duration_to_load:
            y = y[:int(duration_to_load * sr)]
    
    y, _ = librosa.effects.trim(y)
    feats = {}
    
    # ===== EXISTING FEATURES (IMPROVED) =====
    # MFCC (20 instead of 13 - captures more articulation detail)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    feats['mfcc_mean'], feats['mfcc_std'], feats['mfcc_min'], feats['mfcc_max'] = aggregate_stats(mfccs)
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats['chroma_mean'], feats['chroma_std'], feats['chroma_min'], feats['chroma_max'] = aggregate_stats(chroma)
    
    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    feats['mel_mean'], feats['mel_std'], feats['mel_min'], feats['mel_max'] = aggregate_stats(mel)
    
    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    feats['contrast_mean'], feats['contrast_std'], feats['contrast_min'], feats['contrast_max'] = aggregate_stats(contrast)
    
    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    feats['tonnetz_mean'], feats['tonnetz_std'], feats['tonnetz_min'], feats['tonnetz_max'] = aggregate_stats(tonnetz)
    
    # ===== NEW: PROSODY FEATURES (CRITICAL FOR DEPRESSION) =====
    # 1. Pitch (F0) - depression reduces pitch variability
    try:
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_valid = f0[~np.isnan(f0)]
        if len(f0_valid) > 0:
            feats['pitch_mean'] = np.mean(f0_valid)
            feats['pitch_std'] = np.std(f0_valid)
            feats['pitch_range'] = np.max(f0_valid) - np.min(f0_valid)
        else:
            feats['pitch_mean'] = 0
            feats['pitch_std'] = 0
            feats['pitch_range'] = 0
    except:
        feats['pitch_mean'] = 0
        feats['pitch_std'] = 0
        feats['pitch_range'] = 0
    
    # 2. Energy (RMS) - depression reduces vocal energy
    rms = librosa.feature.rms(y=y)[0]
    feats['energy_mean'] = np.mean(rms)
    feats['energy_std'] = np.std(rms)
    feats['energy_range'] = np.max(rms) - np.min(rms)
    
    # 3. Zero Crossing Rate (speech activity)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats['zcr_mean'] = np.mean(zcr)
    feats['zcr_std'] = np.std(zcr)
    
    # 4. Spectral Features (voice quality)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats['centroid_mean'] = np.mean(centroid)
    feats['centroid_std'] = np.std(centroid)
    
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feats['rolloff_mean'] = np.mean(rolloff)
    feats['rolloff_std'] = np.std(rolloff)
    
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    feats['bandwidth_mean'] = np.mean(bandwidth)
    
    # 5. Pause Detection (depression = more pauses/silence)
    intervals = librosa.effects.split(y, top_db=20)
    total_duration = len(y) / sr
    if len(intervals) > 0:
        speech_duration = sum((end - start) / sr for start, end in intervals)
        feats['silence_ratio'] = 1 - (speech_duration / total_duration)
        feats['num_pauses'] = len(intervals) - 1
    else:
        feats['silence_ratio'] = 1.0
        feats['num_pauses'] = 0
    
    # 6. Speech Rate (onset strength)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    feats['onset_mean'] = np.mean(onset_env)
    feats['onset_std'] = np.std(onset_env)
    
    # ✅ Add actual duration used for analysis
    feats['audio_duration'] = len(y) / sr
    
    return feats

def extract_features_from_dataset():
    """
    Extract features from diag-woz dataset
    ✅ EXACTLY FROM YOUR NOTEBOOK
    """
    print(f"📂 Reading dataset from: {DATASET_DIR}")
    
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    
    # Load labels
    labels_df = load_all_labels()
    all_participant_ids = labels_df['Participant_ID'].tolist()
    
    all_features = []
    missing_files = []
    print(f"\n🎵 Starting audio feature extraction for {len(all_participant_ids)} participants...")
    
    for idx, pid in enumerate(tqdm(all_participant_ids, desc="Extracting features"), start=1):
        pid_str = str(pid)
        folder_name = f"{pid_str}_P"
        file_name = f"{pid_str}_AUDIO.wav"
        file_path = f"{DATASET_DIR}/{folder_name}/{folder_name}/{file_name}"
        
        try:
            feats = extract_features_from_file(file_path, sr_target=16000, max_duration=180)
            feats['Participant_ID'] = pid_str
            all_features.append(feats)
        except Exception as e:
            print(f"\nError processing {pid_str}: {e}")
            missing_files.append(file_path)
    
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print('='*60)
    print(f"✅ Extracted features for {len(all_features)} participants")
    print(f"⚠️  Missing/failed files: {len(missing_files)}")
    
    if missing_files and len(missing_files) <= 10:
        print(f"\nMissing files:")
        for mf in missing_files:
            print(f"  - {mf}")
    
    if not all_features:
        raise ValueError("No features extracted! Check your dataset path.")
    
    # Create DataFrame
    df_features = pd.DataFrame(all_features)
    
    # Sort columns to put Participant_ID first
    cols = ['Participant_ID'] + [c for c in df_features.columns if c != 'Participant_ID']
    df_features = df_features[cols]
    
    # Save
    output_path = os.path.join(FEATURES_DIR, "audio_features.csv")
    df_features.to_csv(output_path, index=False)
    
    print(f"\n✅ Features saved to: {output_path}")
    print(f"   Shape: {df_features.shape}")
    print(f"   Columns: {list(df_features.columns)[:10]}...")
    
    # Merge with labels for final dataset
    labels_df = labels_df.rename(columns={'PHQ8_Binary': 'label'})
    df_with_labels = pd.merge(df_features, labels_df[['Participant_ID', 'label']], 
                              on='Participant_ID', how='inner')
    
    print(f"\n   Label distribution:")
    print(df_with_labels['label'].value_counts())
    
    # ✅ Show duration statistics
    if 'audio_duration' in df_features.columns:
        print(f"\n📊 Audio Duration Statistics:")
        print(f"   Mean: {df_features['audio_duration'].mean():.1f}s")
        print(f"   Min:  {df_features['audio_duration'].min():.1f}s")
        print(f"   Max:  {df_features['audio_duration'].max():.1f}s")
    
    return df_features

def load_features():
    """Load features from CSV"""
    features_path = os.path.join(FEATURES_DIR, "audio_features.csv")
    if os.path.exists(features_path):
        print(f"✅ Loading features from: {features_path}")
        df = pd.read_csv(features_path)
        print(f"   Shape: {df.shape}")
        return df
    else:
        raise FileNotFoundError(f"Features file not found at {features_path}")

if __name__ == "__main__":
    print("="*60)
    print("EXTRACTING AUDIO FEATURES FROM DIAG-WOZ DATASET")
    print("="*60)
    
    df = extract_features_from_dataset()
    
    print("\n" + "="*60)
    print("FEATURE NAMES:")
    print("="*60)
    feature_cols = [c for c in df.columns if c not in ['Participant_ID', 'audio_duration']]
    for i, col in enumerate(feature_cols, 1):
        print(f"{i:2d}. {col}")