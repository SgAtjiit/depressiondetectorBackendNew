import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
import joblib
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
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

def compare_text_models():
    print("="*60)
    print("LOADING TEXT FEATURES")
    print("="*60)
    transcripts_df = load_transcripts()
    labels_df = load_all_labels()
    labels_df = labels_df.rename(columns={'PHQ8_Binary': 'label'})
    transcripts_df['Participant_ID'] = transcripts_df['Participant_ID'].astype(str)
    labels_df['Participant_ID'] = labels_df['Participant_ID'].astype(str)
    df = pd.merge(transcripts_df, labels_df[['Participant_ID', 'label']], on='Participant_ID', how='inner')
    X_text = df['transcript']
    y = df['label']
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # TF-IDF
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

    # Feature Selection
    selector = SelectKBest(chi2, k=min(N_FEATURES_TO_SELECT, X_train_tfidf.shape[1]))
    X_train_selected = selector.fit_transform(X_train_tfidf, y_train)
    X_test_selected = selector.transform(X_test_tfidf)

    # SMOTETomek
    smt = SMOTETomek(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smt.fit_resample(X_train_selected.toarray(), y_train)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test_selected.toarray())

    # Models to compare
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)),
        ("SVM (RBF)", SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE)),
        ("Random Forest", RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE)),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE)),
        ("XGBoost", XGBClassifier(
            random_state=RANDOM_STATE,
            scale_pos_weight=len(y_train_resampled[y_train_resampled==0]) / len(y_train_resampled[y_train_resampled==1]),
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            use_label_encoder=False,
            eval_metric='logloss'
        ))
    ]

    # Advanced Ensemble (RF+GB)
    # Hyperparameter tuning for RF
    param_grid = {
        'n_estimators': [200],
        'max_depth': [15],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'max_features': ['sqrt']
    }
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        rf_base, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train_scaled, y_train_resampled)
    best_rf = grid_search.best_estimator_
    gb = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5, min_samples_split=5,
        min_samples_leaf=2, subsample=0.8, random_state=RANDOM_STATE
    )
    ensemble = VotingClassifier(
        estimators=[('rf', best_rf), ('gb', gb)],
        voting='soft', weights=[1.5, 1]
    )
    models.append(("Advanced Ensemble (RF+GB+SMOTETomek+FS)", ensemble))

    # Results
    results = []
    def evaluate_model(name, model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision (1)": report['1']['precision'],
            "Recall (1)": report['1']['recall'],
            "F1-score (1)": report['1']['f1-score'],
            "Precision (0)": report['0']['precision'],
            "Recall (0)": report['0']['recall'],
            "F1-score (0)": report['0']['f1-score'],
            "Support (1)": report['1']['support'],
            "Support (0)": report['0']['support']
        })
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Not Depressed (0)', 'Depressed (1)']))

    # Evaluate all models
    for name, model in models:
        evaluate_model(name, model, X_train_scaled, y_train_resampled, X_test_scaled, y_test)

    # Show results table
    results_df = pd.DataFrame(results)
    print("\n\n==== Model Comparison Table ====")
    print(results_df.sort_values("F1-score (1)", ascending=False).to_string(index=False))

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 STARTING TEXT MODEL TRAINING & COMPARISON")
    print("="*60)
    compare_text_models()
    print("\n" + "="*60)
    print("✅ ALL TRAINING & COMPARISON COMPLETE!")
    print("="*60)