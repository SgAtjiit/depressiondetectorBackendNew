import numpy as np
import pandas as pd
import joblib
import os
import librosa
import soundfile as sf
from pydub import AudioSegment
from io import BytesIO
from config import (
    BEST_MODEL_PATH, 
    SCALER_PATH, 
    SCALER_INITIAL_PATH,
    SELECTOR_PATH,
    MODELS_DIR
)
from feature_extraction import extract_features_from_file

def convert_audio_to_wav(audio_path_or_bytes, output_path="temp_audio.wav"):
    """
    Convert any audio format to WAV (handles webm, mp3, etc.)
    ✅ Works with both file paths and byte streams
    """
    try:
        # Check if input is bytes or file path
        if isinstance(audio_path_or_bytes, (bytes, BytesIO)):
            # Load from bytes
            audio = AudioSegment.from_file(BytesIO(audio_path_or_bytes) if isinstance(audio_path_or_bytes, bytes) else audio_path_or_bytes)
        else:
            # Load from file path
            audio = AudioSegment.from_file(audio_path_or_bytes)
        
        # Convert to WAV
        audio = audio.set_frame_rate(16000).set_channels(1)  # Mono, 16kHz
        audio.export(output_path, format="wav")
        
        print(f"✅ Converted audio to WAV: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"⚠️  Audio conversion failed: {e}")
        return audio_path_or_bytes

def load_model_and_artifacts():
    """Load trained ensemble model, scalers, and feature selector"""
    required_files = {
        'model': BEST_MODEL_PATH,
        'scaler_initial': SCALER_INITIAL_PATH,
        'scaler_final': SCALER_PATH,
        'selector': SELECTOR_PATH
    }
    
    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print("❌ Missing required files:")
        for mf in missing_files:
            print(f"   - {mf}")
        raise FileNotFoundError(
            f"Please train the model first by running: python audio_depression/train.py"
        )
    
    model = joblib.load(BEST_MODEL_PATH)
    scaler_initial = joblib.load(SCALER_INITIAL_PATH)
    scaler_final = joblib.load(SCALER_PATH)
    selector = joblib.load(SELECTOR_PATH)
    
    return model, scaler_initial, scaler_final, selector

def predict_depression(audio_path_or_bytes):
    """
    Predict depression from audio file or bytes
    ✅ Automatically handles variable duration audio
    ✅ Converts any format to WAV
    Returns: dict with prediction, probabilities, and confidence
    """
    temp_file = None
    
    try:
        # ✅ Convert audio to WAV if needed
        if isinstance(audio_path_or_bytes, (bytes, BytesIO)):
            print("🔄 Processing audio bytes...")
            temp_file = "temp_audio.wav"
            audio_path = convert_audio_to_wav(audio_path_or_bytes, temp_file)
        else:
            if not os.path.exists(audio_path_or_bytes):
                raise FileNotFoundError(f"Audio file not found: {audio_path_or_bytes}")
            
            # Check if conversion is needed
            if not audio_path_or_bytes.lower().endswith('.wav'):
                print(f"🔄 Converting {audio_path_or_bytes} to WAV...")
                temp_file = "temp_audio_converted.wav"
                audio_path = convert_audio_to_wav(audio_path_or_bytes, temp_file)
            else:
                audio_path = audio_path_or_bytes
        
        # Load model artifacts
        model, scaler_initial, scaler_final, selector = load_model_and_artifacts()
        
        # ✅ Get audio duration using soundfile (more reliable)
        try:
            audio_info = sf.info(audio_path)
            audio_duration = audio_info.duration
            print(f"🎵 Audio duration: {audio_duration:.1f}s (sample rate: {audio_info.samplerate} Hz)")
        except Exception as e:
            print(f"⚠️  Using librosa for duration: {e}")
            audio_duration = librosa.get_duration(path=audio_path)
            print(f"🎵 Audio duration: {audio_duration:.1f}s")
        
        # Extract features (automatically handles duration)
        print(f"🔬 Extracting features...")
        features_dict = extract_features_from_file(audio_path, sr_target=16000, max_duration=180)
        
        # Remove audio_duration if present (not a feature for model)
        actual_duration = features_dict.pop('audio_duration', audio_duration)
        print(f"✅ Analyzed duration: {actual_duration:.1f}s")
        
        features_df = pd.DataFrame([features_dict])
        
        # Apply pipeline
        features_scaled = scaler_initial.transform(features_df)
        features_selected = selector.transform(features_scaled)
        features_final = scaler_final.transform(features_selected)
        
        # Predict
        prediction = model.predict(features_final)[0]
        probability = model.predict_proba(features_final)[0]
        
        result = {
            'prediction': int(prediction),
            'label': 'Depressed' if prediction == 1 else 'Not Depressed',
            'probability': {
                'not_depressed': float(probability[0]),
                'depressed': float(probability[1])
            },
            'confidence': float(max(probability)),
            'audio_duration': float(audio_duration),
            'analyzed_duration': float(actual_duration)
        }
        
        print(f"✅ Prediction complete: {result['label']} ({result['confidence']:.2%} confidence)")
        
        return result
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"🗑️  Cleaned up: {temp_file}")
            except:
                pass

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        
        try:
            result = predict_depression(audio_path)
            
            print("\n" + "="*60)
            print(f"🎯 Prediction: {result['label']}")
            print(f"📊 Confidence: {result['confidence']:.2%}")
            print(f"\n📈 Probabilities:")
            print(f"   Not Depressed: {result['probability']['not_depressed']:.2%}")
            print(f"   Depressed:     {result['probability']['depressed']:.2%}")
            print(f"\n⏱️  Duration Info:")
            print(f"   Original audio: {result['audio_duration']:.1f}s")
            print(f"   Analyzed:       {result['analyzed_duration']:.1f}s")
            print("="*60)
        
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            sys.exit(1)
    else:
        print("Usage: python predict.py <path_to_audio_file>")
        print("\nExample:")
        print('  python predict.py "D:/depressiondetector/test_audio.wav"')