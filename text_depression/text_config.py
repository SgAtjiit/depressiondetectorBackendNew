import os

# ===== PROJECT ROOT =====
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ===== EXTERNAL DATA (Outside project folder) =====
DATASET_DIR = r"D:\depressiondetector\diag-woz"
LABELS_DIR = r"D:\depressiondetector\labels"

# Label files
TRAIN_LABELS_FILE = os.path.join(LABELS_DIR, "train_split.csv")
DEV_LABELS_FILE = os.path.join(LABELS_DIR, "dev_split.csv")
TEST_LABELS_FILE = os.path.join(LABELS_DIR, "test_split.csv")

# ===== INTERNAL DATA (Inside project folder) =====
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# ===== MODELS (Inside project folder) =====
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_text_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "text_vectorizer.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "text_scaler.pkl")

# ===== RESULTS (Inside project folder) =====
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

# ===== TEXT PROCESSING PARAMETERS =====
MAX_FEATURES = 5000  # Maximum number of features for TfidfVectorizer
MIN_DF = 2           # Minimum document frequency
MAX_DF = 0.95        # Maximum document frequency (ignore too common words)
NGRAM_RANGE = (1, 2) # Use unigrams and bigrams
MAX_SEQUENCE_LENGTH = 500
EMBEDDING_DIM = 100

# ===== TRAINING PARAMETERS =====
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
BATCH_SIZE = 32
EPOCHS = 50

# ===== FEATURE SELECTION =====
N_FEATURES_TO_SELECT = 1000  # Number of features to select using SelectKBest

# ===== CREATE DIRECTORIES =====
def create_directories():
    """Create all necessary directories inside the project folder"""
    directories = [
        DATA_DIR,
        FEATURES_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        PLOTS_DIR,
        METRICS_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print(f"✅ All directories created inside: {PROJECT_ROOT}")
    print(f"✅ Dataset location (external): {DATASET_DIR}")
    print(f"✅ Labels location (external): {LABELS_DIR}")

# Create directories on import
create_directories()