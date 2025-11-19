import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import joblib
import os
from text_config import (
    FEATURES_DIR,
    MODELS_DIR,
    BEST_MODEL_PATH,
    VECTORIZER_PATH,
    SCALER_PATH,
    RANDOM_STATE,
    MAX_FEATURES,
    MIN_DF,
    MAX_DF,
    NGRAM_RANGE,
    N_FEATURES_TO_SELECT,
    TEST_SIZE,
    CV_FOLDS
)
from text_feature_extraction import load_transcripts, load_all_labels

def train_text_model():
    """
    Train text-based depression detection model
    ✅ TF-IDF → Feature Selection → SMOTETomek → GridSearch → Ensemble
    """
    print("="*60)
    print("LOADING TEXT FEATURES")
    print("="*60)
    
    # Load transcripts
    transcripts_df = load_transcripts()
    
    # Load labels
    labels_df = load_all_labels()
    labels_df = labels_df.rename(columns={'PHQ8_Binary': 'label'})
    
    # Convert to string for merging
    transcripts_df['Participant_ID'] = transcripts_df['Participant_ID'].astype(str)
    labels_df['Participant_ID'] = labels_df['Participant_ID'].astype(str)
    
    # Merge
    df = pd.merge(transcripts_df, labels_df[['Participant_ID', 'label']], 
                  on='Participant_ID', how='inner')
    
    print(f"✅ Merged dataset shape: {df.shape}")
    print(f"   Class distribution:\n{df['label'].value_counts()}")
    
    # Prepare data
    X_text = df['transcript']
    y = df['label']
    
    # Split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train_text)} samples")
    print(f"Test set: {len(X_test_text)} samples")
    
    # ===== STEP 1: TF-IDF Vectorization =====
    print("\n" + "="*60)
    print("STEP 1: TF-IDF VECTORIZATION")
    print("="*60)
    
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=NGRAM_RANGE,
        stop_words='english',
        strip_accents='unicode',
        lowercase=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    
    print(f"TF-IDF features: {X_train_tfidf.shape[1]}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # ===== STEP 2: Feature Selection =====
    print("\n" + "="*60)
    print(f"STEP 2: FEATURE SELECTION (Top {N_FEATURES_TO_SELECT})")
    print("="*60)
    
    selector = SelectKBest(chi2, k=min(N_FEATURES_TO_SELECT, X_train_tfidf.shape[1]))
    X_train_selected = selector.fit_transform(X_train_tfidf, y_train)
    X_test_selected = selector.transform(X_test_tfidf)
    
    print(f"Original features: {X_train_tfidf.shape[1]}")
    print(f"Selected features: {X_train_selected.shape[1]}")
    
    # Get selected feature names
    feature_names = vectorizer.get_feature_names_out()
    selected_features = feature_names[selector.get_support()]
    print(f"\nTop 20 selected features: {list(selected_features[:20])}")
    
    # ===== STEP 3: SMOTETomek =====
    print("\n" + "="*60)
    print("STEP 3: APPLYING SMOTETomek")
    print("="*60)
    
    smt = SMOTETomek(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smt.fit_resample(X_train_selected.toarray(), y_train)
    
    print(f"Before SMOTETomek: {dict(y_train.value_counts())}")
    print(f"After SMOTETomek:  {dict(pd.Series(y_train_resampled).value_counts())}")
    
    # ===== STEP 4: Scaling =====
    print("\n" + "="*60)
    print("STEP 4: SCALING FEATURES")
    print("="*60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test_selected.toarray())
    
    # ===== STEP 5: Hyperparameter Tuning =====
    print("\n" + "="*60)
    print("STEP 5: HYPERPARAMETER TUNING (Random Forest)")
    print("="*60)
    
    param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    rf_base = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    )
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        rf_base, 
        param_grid, 
        cv=cv, 
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    print("Starting Grid Search (this may take a few minutes)...")
    grid_search.fit(X_train_scaled, y_train_resampled)
    
    print(f"\n✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV F1 score: {grid_search.best_score_:.4f}")
    
    # ===== STEP 6: Train Ensemble Model =====
    print("\n" + "="*60)
    print("STEP 6: TRAINING ENSEMBLE MODEL")
    print("="*60)
    
    best_rf = grid_search.best_estimator_
    
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=RANDOM_STATE
    )
    
    ensemble = VotingClassifier(
        estimators=[
            ('rf', best_rf),
            ('gb', gb)
        ],
        voting='soft',
        weights=[1.5, 1]
    )
    
    print("Training ensemble model...")
    ensemble.fit(X_train_scaled, y_train_resampled)
    
    # ===== STEP 7: Evaluate =====
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)
    
    y_pred = ensemble.predict(X_test_scaled)
    y_pred_proba = ensemble.predict_proba(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✅ Weighted F1 Score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Not Depressed (0)', 'Depressed (1)']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print(f"\nSpecificity (Class 0): {specificity:.4f}")
    print(f"Sensitivity (Class 1): {sensitivity:.4f}")
    
    # ===== STEP 8: Save Everything =====
    print("\n" + "="*60)
    print("SAVING MODEL, VECTORIZER, SCALER, AND SELECTOR")
    print("="*60)
    
    joblib.dump(ensemble, BEST_MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    selector_path = os.path.join(MODELS_DIR, "feature_selector.pkl")
    joblib.dump(selector, selector_path)
    
    print(f"✅ Ensemble model saved to: {BEST_MODEL_PATH}")
    print(f"✅ Vectorizer saved to: {VECTORIZER_PATH}")
    print(f"✅ Scaler saved to: {SCALER_PATH}")
    print(f"✅ Feature selector saved to: {selector_path}")
    print(f"\n🎯 Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Feature importances
    rf_model = ensemble.estimators_[0]
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Most Important Words/Phrases:")
    print(feature_importance.head(20).to_string(index=False))
    
    return ensemble, vectorizer, scaler, selector

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 STARTING TEXT MODEL TRAINING")
    print("="*60)
    
    model, vectorizer, scaler, selector = train_text_model()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)