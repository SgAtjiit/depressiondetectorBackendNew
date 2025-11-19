import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from text_config import (
    DATASET_DIR,
    LABELS_DIR,
    TRAIN_LABELS_FILE,
    DEV_LABELS_FILE,
    TEST_LABELS_FILE,
    FEATURES_DIR,
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

def clean_text(text):
    """
    Deep clean transcript text - remove ALL noise
    ✅ Remove timestamps (2.1,3.2,)
    ✅ Remove confidence scores (0.9876289963722229)
    ✅ Remove standalone numbers
    ✅ Remove special characters
    ✅ Convert to lowercase
    ✅ Remove extra whitespace
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Step 1: Remove timestamp pairs (e.g., "2.1,3.2,")
    text = re.sub(r'\d+\.\d+,\s*\d+\.\d+,', '', text)
    
    # Step 2: Remove confidence scores (e.g., "0.9876289963722229")
    text = re.sub(r',\s*0\.\d+', '', text)
    
    # Step 3: Remove standalone timestamps (e.g., "2.1,")
    text = re.sub(r'\d+\.\d+,', '', text)
    
    # Step 4: Remove standalone numbers at word boundaries
    text = re.sub(r'\b\d+\b', '', text)
    
    # Step 5: Remove extra commas
    text = re.sub(r',+', ',', text)
    text = text.strip(',')
    
    # Step 6: Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z\s\.\,\?\!\'\-]', '', text)
    
    # Step 7: Convert to lowercase
    text = text.lower()
    
    # Step 8: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Step 9: Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def extract_transcript_from_csv(file_path):
    """
    Extract and clean text from DAIC-WOZ transcript CSV
    Format: start_time, stop_time, text, confidence
    ✅ Skips header row
    ✅ Extracts only the text column
    """
    try:
        # Read transcript file (tab-separated, skip header)
        df = pd.read_csv(file_path, sep='\t', header=0, names=['start_time', 'stop_time', 'text', 'confidence'])
        
        # Extract text column (3rd column)
        if 'text' in df.columns:
            texts = df['text'].dropna().astype(str).tolist()
        elif df.shape[1] >= 3:
            # Fallback: use 3rd column (index 2)
            texts = df.iloc[:, 2].dropna().astype(str).tolist()
        else:
            # Last resort: take all text
            texts = df.values.flatten()
        
        # Combine all text
        full_text = ' '.join(texts)
        
        # Clean the text
        cleaned_text = clean_text(full_text)
        
        return cleaned_text
    
    except Exception as e:
        # Try alternative reading method (no header)
        try:
            df = pd.read_csv(file_path, sep='\t', header=None)
            
            # Skip first row if it looks like a header
            if df.shape[0] > 0:
                first_row = df.iloc[0].astype(str).str.lower()
                if any('starttime' in str(val) or 'text' in str(val) for val in first_row):
                    df = df.iloc[1:]  # Skip header row
            
            # Extract text (3rd column, index 2)
            if df.shape[1] >= 3:
                texts = df.iloc[:, 2].dropna().astype(str).tolist()
            else:
                texts = df.values.flatten()
            
            full_text = ' '.join(texts)
            cleaned_text = clean_text(full_text)
            
            return cleaned_text
        
        except Exception as e2:
            raise Exception(f"Error reading transcript from {file_path}: {e2}")

def extract_text_features():
    """
    Extract and clean text transcripts from diag-woz dataset
    ✅ Proper CSV parsing with header handling
    ✅ Deep text cleaning (remove timestamps, confidence, noise)
    """
    print(f"📂 Reading dataset from: {DATASET_DIR}")
    
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    
    # Load labels
    labels_df = load_all_labels()
    all_participant_ids = labels_df['Participant_ID'].tolist()
    
    all_transcripts = []
    missing_files = []
    empty_transcripts = []
    
    print(f"\n📝 Starting text extraction for {len(all_participant_ids)} participants...")
    print("   (Deep cleaning: removing timestamps, confidence scores, all noise)\n")
    
    for pid in tqdm(all_participant_ids, desc="Extracting & cleaning"):
        pid_str = str(pid)
        folder_name = f"{pid_str}_P"
        file_name = f"{pid_str}_TRANSCRIPT.csv"
        file_path = os.path.join(DATASET_DIR, folder_name, folder_name, file_name)
        
        try:
            # Extract and clean transcript
            cleaned_text = extract_transcript_from_csv(file_path)
            
            if cleaned_text and len(cleaned_text) > 10:  # At least 10 chars
                all_transcripts.append({
                    'Participant_ID': pid_str,
                    'transcript': cleaned_text
                })
            else:
                empty_transcripts.append(pid_str)
            
        except Exception as e:
            print(f"\n⚠️  Error processing {pid_str}: {e}")
            missing_files.append(file_path)
    
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print('='*60)
    print(f"✅ Extracted transcripts: {len(all_transcripts)}")
    print(f"⚠️  Missing files: {len(missing_files)}")
    print(f"⚠️  Empty transcripts: {len(empty_transcripts)}")
    
    if missing_files and len(missing_files) <= 10:
        print(f"\nMissing files (first 10):")
        for mf in missing_files[:10]:
            print(f"  - {mf}")
    
    if empty_transcripts and len(empty_transcripts) <= 10:
        print(f"\nEmpty transcripts (first 10):")
        for et in empty_transcripts[:10]:
            print(f"  - Participant {et}")
    
    if not all_transcripts:
        raise ValueError("No valid transcripts extracted! Check your dataset path.")
    
    # Create DataFrame
    df_transcripts = pd.DataFrame(all_transcripts)
    
    # Save
    output_path = os.path.join(FEATURES_DIR, "text_transcripts.csv")
    df_transcripts.to_csv(output_path, index=False)
    
    print(f"\n✅ Cleaned transcripts saved to: {output_path}")
    print(f"   Shape: {df_transcripts.shape}")
    print(f"   Average length: {df_transcripts['transcript'].str.len().mean():.0f} characters")
    print(f"   Min length: {df_transcripts['transcript'].str.len().min()}")
    print(f"   Max length: {df_transcripts['transcript'].str.len().max()}")
    
    # Merge with labels
    labels_df = labels_df.rename(columns={'PHQ8_Binary': 'label'})
    df_with_labels = pd.merge(df_transcripts, labels_df[['Participant_ID', 'label']], 
                              on='Participant_ID', how='inner')
    
    print(f"\n   Final dataset shape: {df_with_labels.shape}")
    print(f"   Label distribution:")
    print(df_with_labels['label'].value_counts())
    
    return df_transcripts

def load_transcripts():
    """Load transcripts from CSV"""
    transcripts_path = os.path.join(FEATURES_DIR, "text_transcripts.csv")
    if os.path.exists(transcripts_path):
        print(f"✅ Loading transcripts from: {transcripts_path}")
        df = pd.read_csv(transcripts_path)
        df['Participant_ID'] = df['Participant_ID'].astype(str)
        print(f"   Shape: {df.shape}")
        return df
    else:
        raise FileNotFoundError(f"Transcripts file not found at {transcripts_path}")

if __name__ == "__main__":
    print("="*60)
    print("EXTRACTING & DEEP CLEANING TEXT TRANSCRIPTS")
    print("="*60)
    
    df = extract_text_features()
    
    print("\n" + "="*60)
    print("SAMPLE CLEANED TRANSCRIPTS:")
    print("="*60)
    
    # Show first 3 transcripts (truncated)
    for idx, row in df.head(3).iterrows():
        pid = row['Participant_ID']
        text = row['transcript']
        preview = text[:300] + "..." if len(text) > 300 else text
        print(f"\n[Participant {pid}]")
        print(f"{preview}\n")
    
    print("\n" + "="*60)
    print("✅ EXTRACTION COMPLETE!")
    print("="*60)