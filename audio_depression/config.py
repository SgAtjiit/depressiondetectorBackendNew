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
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_audio_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.pkl")
SCALER_INITIAL_PATH = os.path.join(MODELS_DIR, "scaler_initial.pkl")  # ✅ ADDED
SELECTOR_PATH = os.path.join(MODELS_DIR, "feature_selector.pkl")      # ✅ ADDED

# ===== RESULTS (Inside project folder) =====
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

# ===== AUDIO PARAMETERS =====
SAMPLE_RATE = 16000
N_MFCC = 20
N_MELS = 128
FRAME_LENGTH = 2048
HOP_LENGTH = 512
MAX_DURATION = 180  # 3 minutes

# ===== TRAINING PARAMETERS =====
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

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
    
    print(f"✅ Directories created inside: {PROJECT_ROOT}")
    print(f"✅ Dataset location: {DATASET_DIR}")
    print(f"✅ Labels location: {LABELS_DIR}")

# Create directories on import
create_directories()