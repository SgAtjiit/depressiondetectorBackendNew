import numpy as np
import pandas as pd
import joblib
import os
from text_config import (
    BEST_MODEL_PATH,
    VECTORIZER_PATH,
    SCALER_PATH,
    MODELS_DIR
)

def load_text_model_and_artifacts():
    """Load trained text model, vectorizer, scaler, and selector"""
    model_path = BEST_MODEL_PATH
    vectorizer_path = VECTORIZER_PATH
    scaler_path = SCALER_PATH
    selector_path = os.path.join(MODELS_DIR, "feature_selector.pkl")
    
    required_files = {
        'model': model_path,
        'vectorizer': vectorizer_path,
        'scaler': scaler_path,
        'selector': selector_path
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
            f"Please train the model first by running: python text_depression/text_train.py"
        )
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    scaler = joblib.load(scaler_path)
    selector = joblib.load(selector_path)
    
    return model, vectorizer, scaler, selector

def extract_linguistic_features(text):
    """Extract simple linguistic features for interpretation"""
    words = text.split()
    sentences = text.split('.')
    
    # Depression-related keywords
    negative_words = ['sad', 'depressed', 'hopeless', 'alone', 'tired', 'empty', 'worthless', 'anxious', 'nervous', 'worry']
    positive_words = ['happy', 'good', 'great', 'excited', 'wonderful', 'love', 'enjoy', 'fun', 'glad', 'pleased']
    
    negative_count = sum(1 for word in words if word.lower() in negative_words)
    positive_count = sum(1 for word in words if word.lower() in positive_words)
    
    return {
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'negative_word_count': negative_count,
        'positive_word_count': positive_count,
        'sentiment_ratio': (positive_count - negative_count) / max(len(words), 1)
    }

def predict_depression_from_text(text):
    """
    Predict depression from text input
    Returns: dict with prediction, probabilities, confidence, and features
    """
    if not text or len(text.strip()) < 10:
        raise ValueError("Text is too short. Please provide at least 10 characters.")
    
    model, vectorizer, scaler, selector = load_text_model_and_artifacts()
    
    # Apply pipeline
    text_tfidf = vectorizer.transform([text])
    text_selected = selector.transform(text_tfidf)
    text_scaled = scaler.transform(text_selected.toarray())
    
    # Predict
    prediction = model.predict(text_scaled)[0]
    probability = model.predict_proba(text_scaled)[0]
    
    # Extract linguistic features
    ling_features = extract_linguistic_features(text)
    
    result = {
        'prediction': int(prediction),
        'label': 'Depressed' if prediction == 1 else 'Not Depressed',
        'probability': {
            'not_depressed': float(probability[0]),
            'depressed': float(probability[1])
        },
        'confidence': float(max(probability)),
        'linguistic_features': ling_features
    }
    
    return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        text_input = ' '.join(sys.argv[1:])
        
        try:
            result = predict_depression_from_text(text_input)
            
            print("\n" + "="*60)
            print(f"🎯 Prediction: {result['label']}")
            print(f"📊 Confidence: {result['confidence']:.2%}")
            print(f"\n📈 Probabilities:")
            print(f"   Not Depressed: {result['probability']['not_depressed']:.2%}")
            print(f"   Depressed:     {result['probability']['depressed']:.2%}")
            print(f"\n📝 Linguistic Features:")
            for key, val in result['linguistic_features'].items():
                print(f"   {key}: {val}")
            print("="*60)
        
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            sys.exit(1)
    else:
        print("Usage: python text_predict.py <text>")
        print("\nExample:")
        print('  python text_predict.py "I feel very sad and hopeless today"')