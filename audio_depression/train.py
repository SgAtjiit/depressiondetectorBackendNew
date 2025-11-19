import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import joblib
import os
from config import (
    FEATURES_DIR,
    MODELS_DIR,
    BEST_MODEL_PATH,
    SCALER_PATH,
    RANDOM_STATE
)
from feature_extraction import load_features, load_all_labels

def train_model():
    """
    Train model with ADVANCED TECHNIQUES for 75%+ accuracy
    ✅ Better resampling (SMOTETomek)
    ✅ Hyperparameter tuning (GridSearchCV)
    ✅ Ensemble methods (Voting Classifier)
    ✅ More features (k=30 instead of 25)
    """
    print("="*60)
    print("LOADING FEATURES")
    print("="*60)
    
    # Load features
    features_df = load_features()
    
    # Load labels
    labels_df = load_all_labels()
    labels_df = labels_df.rename(columns={'PHQ8_Binary': 'label'})
    
    # Convert to string for merging
    features_df['Participant_ID'] = features_df['Participant_ID'].astype(str)
    labels_df['Participant_ID'] = labels_df['Participant_ID'].astype(str)
    
    # Merge
    df = pd.merge(features_df, labels_df[['Participant_ID', 'label']], 
                  on='Participant_ID', how='inner')
    
    print(f"✅ Merged dataset shape: {df.shape}")
    
    # Separate features
    feature_cols = [c for c in df.columns if c not in ['Participant_ID', 'label']]
    X = df[feature_cols]
    y = df['label']
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Class distribution:\n{y.value_counts()}")
    
    # Split (use stratified split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # ===== STEP 1: Scale Original Data =====
    print("\n" + "="*60)
    print("STEP 1: INITIAL SCALING")
    print("="*60)
    scaler_initial = StandardScaler()
    X_train_scaled = scaler_initial.fit_transform(X_train)
    X_test_scaled = scaler_initial.transform(X_test)
    
    # ===== STEP 2: Feature Selection (MORE features: k=30) =====
    print("\n" + "="*60)
    print("STEP 2: FEATURE SELECTION (Top 30)")
    print("="*60)
    
    # Use more features (30 instead of 25)
    k_features = min(30, X_train.shape[1])
    selector = SelectKBest(f_classif, k=k_features)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Selected features: {X_train_selected.shape[1]}")
    
    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
    print(f"\nSelected features: {selected_features[:10]}...")
    
    # ===== STEP 3: SMOTETomek (Better than SMOTE alone) =====
    print("\n" + "="*60)
    print("STEP 3: APPLYING SMOTETomek (SMOTE + Tomek Links)")
    print("="*60)
    
    # SMOTETomek = SMOTE + removes noisy samples
    smt = SMOTETomek(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smt.fit_resample(X_train_selected, y_train)
    
    print(f"Before SMOTETomek: {dict(y_train.value_counts())}")
    print(f"After SMOTETomek:  {dict(pd.Series(y_train_resampled).value_counts())}")
    
    # ===== STEP 4: Scale Resampled Data =====
    print("\n" + "="*60)
    print("STEP 4: SCALING RESAMPLED DATA")
    print("="*60)
    scaler_final = StandardScaler()
    X_train_final = scaler_final.fit_transform(X_train_resampled)
    X_test_final = scaler_final.transform(X_test_selected)
    
    # ===== STEP 5: Hyperparameter Tuning with GridSearchCV =====
    print("\n" + "="*60)
    print("STEP 5: HYPERPARAMETER TUNING (Random Forest)")
    print("="*60)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    # Base model
    rf_base = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    )
    
    # Grid search with cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        rf_base, 
        param_grid, 
        cv=cv, 
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    print("Starting Grid Search (this may take a few minutes)...")
    grid_search.fit(X_train_final, y_train_resampled)
    
    print(f"\n✅ Best parameters found: {grid_search.best_params_}")
    print(f"✅ Best cross-validation F1 score: {grid_search.best_score_:.4f}")
    
    # ===== STEP 6: Train Ensemble Model =====
    print("\n" + "="*60)
    print("STEP 6: TRAINING ENSEMBLE MODEL")
    print("="*60)
    
    # Use best Random Forest
    best_rf = grid_search.best_estimator_
    
    # Create Gradient Boosting model
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=RANDOM_STATE
    )
    
    # Ensemble: Voting Classifier (RF + GB)
    ensemble = VotingClassifier(
        estimators=[
            ('rf', best_rf),
            ('gb', gb)
        ],
        voting='soft',  # Use probability voting
        weights=[1.5, 1]  # Give more weight to RF
    )
    
    print("Training ensemble model (Random Forest + Gradient Boosting)...")
    ensemble.fit(X_train_final, y_train_resampled)
    
    # ===== STEP 7: Evaluate =====
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)
    
    y_pred = ensemble.predict(X_test_final)
    y_pred_proba = ensemble.predict_proba(X_test_final)
    
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
    
    # Calculate per-class accuracy
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print(f"\nSpecificity (Class 0): {specificity:.4f}")
    print(f"Sensitivity (Class 1): {sensitivity:.4f}")
    
    # ===== STEP 8: Save Everything =====
    print("\n" + "="*60)
    print("SAVING MODEL, SCALERS, AND SELECTOR")
    print("="*60)
    
    # Save ensemble model
    joblib.dump(ensemble, BEST_MODEL_PATH)
    
    # Save scalers
    scaler_initial_path = os.path.join(MODELS_DIR, "scaler_initial.pkl")
    joblib.dump(scaler_initial, scaler_initial_path)
    joblib.dump(scaler_final, SCALER_PATH)
    
    # Save feature selector
    selector_path = os.path.join(MODELS_DIR, "feature_selector.pkl")
    joblib.dump(selector, selector_path)
    
    print(f"✅ Ensemble model saved to: {BEST_MODEL_PATH}")
    print(f"✅ Initial scaler saved to: {scaler_initial_path}")
    print(f"✅ Final scaler saved to: {SCALER_PATH}")
    print(f"✅ Feature selector saved to: {selector_path}")
    print(f"\n🎯 Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Show feature importances from Random Forest
    rf_model = ensemble.estimators_[0]  # Get RF from ensemble
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15).to_string(index=False))
    
    return ensemble, scaler_initial, scaler_final, selector

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 STARTING ADVANCED MODEL TRAINING")
    print("="*60)
    
    model, scaler_initial, scaler_final, selector = train_model()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)