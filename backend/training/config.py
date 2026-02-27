"""
Training Configuration — Hyperparameters & Paths.
==================================================
Tập trung toàn bộ cấu hình huấn luyện vào 1 file.
"""
from pathlib import Path

# ========================
# PATHS
# ========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # → nlp/
BACKEND_DIR = PROJECT_ROOT / "backend"

# Data sources
TEST_DATASET_PATH = BACKEND_DIR / "evaluation" / "test_dataset.json"

# Model output
TFIDF_MODEL_DIR = BACKEND_DIR / "resources" / "tfidf"

# ========================
# TF-IDF VECTORIZER
# ========================
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)       # unigram + bigram
TFIDF_SUBLINEAR_TF = True        # 1 + log(tf) — giảm ảnh hưởng từ lặp nhiều
TFIDF_MIN_DF = 1                 # min document frequency
TFIDF_MAX_DF = 0.95              # max document frequency (bỏ stopwords tự nhiên)

# ========================
# CLASSIFIER
# ========================
CLASSIFIER_TYPE = "logistic_regression"  # "logistic_regression" | "svm"
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 1000
SVM_C = 1.0

# ========================
# TRAINING
# ========================
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_F1_THRESHOLD = 0.70  # Ngưỡng F1 tối thiểu để chấp nhận model

# ========================
# DATA AUGMENTATION
# ========================
AUGMENT_ENABLED = True
AUGMENT_MULTIPLIER = 2  # Mỗi sample gốc tạo thêm N sample augmented

# ========================
# VERSIONING
# ========================
MODEL_VERSION = "1.0.0"
