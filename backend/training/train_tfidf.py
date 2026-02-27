"""
TF-IDF Training Pipeline — Main Script
========================================
Pipeline 6 bước:
  1. Thu thập & chuẩn hóa dữ liệu (3 sources)
  2. Train TF-IDF Vectorizer
  3. Train Classifier (Logistic Regression + SVM, chọn tốt hơn)
  4. Đánh giá (accuracy, F1, confusion matrix)
  5. Lưu model artifacts (.pkl + metadata)
  6. Versioning

Usage:
  cd /mnt/atin/QuyNB/project/master_project/nlp
  conda run -n mlops python -m backend.training.train_tfidf
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.config import (
    CLASSIFIER_TYPE,
    LOGISTIC_C,
    LOGISTIC_MAX_ITER,
    MIN_F1_THRESHOLD,
    MODEL_VERSION,
    RANDOM_STATE,
    SVM_C,
    TEST_SIZE,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TFIDF_MODEL_DIR,
    TFIDF_SUBLINEAR_TF,
)
from backend.training.data_preparation import prepare_training_data


def main():
    print("=" * 60)
    print("  🚀 TF-IDF Training Pipeline")
    print(f"  Version: {MODEL_VERSION}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    t_start = time.perf_counter()

    # ============================================================
    # STEP 1: Thu thập & chuẩn hóa dữ liệu
    # ============================================================
    texts, labels = prepare_training_data()

    # ============================================================
    # STEP 2: Train TF-IDF Vectorizer
    # ============================================================
    print("\n🔧 Step 2: Training TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
    )

    X = vectorizer.fit_transform(texts)
    print(f"  ✅ Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  ✅ Feature matrix: {X.shape}")

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    print(f"  ✅ Classes: {list(label_encoder.classes_)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  ✅ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ============================================================
    # STEP 3: Train Classifier(s)
    # ============================================================
    print("\n🔧 Step 3: Training Classifiers...")

    # Train cả 2 classifier, chọn tốt hơn
    classifiers = {
        "logistic_regression": LogisticRegression(
            C=LOGISTIC_C,
            max_iter=LOGISTIC_MAX_ITER,
            random_state=RANDOM_STATE,
            multi_class="multinomial",
            solver="lbfgs",
        ),
        "svm": LinearSVC(
            C=SVM_C,
            random_state=RANDOM_STATE,
            max_iter=LOGISTIC_MAX_ITER,
        ),
    }

    best_clf_name = None
    best_clf = None
    best_f1 = 0.0
    clf_results = {}

    for clf_name, clf in classifiers.items():
        print(f"\n  Training {clf_name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        # Cross-validation (5-fold)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")

        clf_results[clf_name] = {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1, 4),
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std": round(cv_scores.std(), 4),
        }

        print(f"    Accuracy: {acc:.4f}")
        print(f"    F1 Macro: {f1:.4f}")
        print(f"    CV F1:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_clf_name = clf_name
            best_clf = clf

    print(f"\n  🏆 Best classifier: {best_clf_name} (F1={best_f1:.4f})")

    # ============================================================
    # STEP 4: Đánh giá chi tiết
    # ============================================================
    print("\n📊 Step 4: Evaluation...")

    y_pred_best = best_clf.predict(X_test)
    report = classification_report(
        y_test, y_pred_best,
        target_names=label_encoder.classes_,
        output_dict=True,
    )
    report_text = classification_report(
        y_test, y_pred_best,
        target_names=label_encoder.classes_,
    )
    cm = confusion_matrix(y_test, y_pred_best)

    print(report_text)

    # Check F1 threshold
    final_f1 = report["macro avg"]["f1-score"]
    final_acc = report["accuracy"]

    if final_f1 < MIN_F1_THRESHOLD:
        print(f"\n  ⚠️  WARNING: F1={final_f1:.4f} < threshold={MIN_F1_THRESHOLD}")
        print("     Model chưa đạt ngưỡng chất lượng.")
    else:
        print(f"\n  ✅ F1={final_f1:.4f} ≥ threshold={MIN_F1_THRESHOLD} — PASSED")

    # ============================================================
    # STEP 5: Lưu model artifacts
    # ============================================================
    print("\n💾 Step 5: Saving model artifacts...")

    TFIDF_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save vectorizer
    vectorizer_path = TFIDF_MODEL_DIR / "vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_path)
    print(f"  ✅ {vectorizer_path}")

    # Save classifier
    classifier_path = TFIDF_MODEL_DIR / "classifier.pkl"
    joblib.dump(best_clf, classifier_path)
    print(f"  ✅ {classifier_path}")

    # Save label encoder
    encoder_path = TFIDF_MODEL_DIR / "label_encoder.pkl"
    joblib.dump(label_encoder, encoder_path)
    print(f"  ✅ {encoder_path}")

    # ============================================================
    # STEP 6: Metadata & Versioning
    # ============================================================
    print("\n📋 Step 6: Saving metadata & version info...")

    elapsed = time.perf_counter() - t_start

    metadata = {
        "version": MODEL_VERSION,
        "trained_at": datetime.now().isoformat(),
        "training_time_seconds": round(elapsed, 2),
        "classifier": best_clf_name,
        "classifier_comparison": clf_results,
        "tfidf_config": {
            "max_features": TFIDF_MAX_FEATURES,
            "ngram_range": list(TFIDF_NGRAM_RANGE),
            "sublinear_tf": TFIDF_SUBLINEAR_TF,
            "min_df": TFIDF_MIN_DF,
            "max_df": TFIDF_MAX_DF,
        },
        "data": {
            "total_samples": len(texts),
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "vocabulary_size": len(vectorizer.vocabulary_),
            "num_classes": len(label_encoder.classes_),
            "classes": list(label_encoder.classes_),
        },
        "evaluation": {
            "accuracy": round(final_acc, 4),
            "f1_macro": round(final_f1, 4),
            "per_class": {
                cls: {
                    "precision": round(report[cls]["precision"], 4),
                    "recall": round(report[cls]["recall"], 4),
                    "f1-score": round(report[cls]["f1-score"], 4),
                    "support": report[cls]["support"],
                }
                for cls in label_encoder.classes_
            },
            "confusion_matrix": cm.tolist(),
        },
    }

    metadata_path = TFIDF_MODEL_DIR / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {metadata_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  ✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Version:     {MODEL_VERSION}")
    print(f"  Classifier:  {best_clf_name}")
    print(f"  Accuracy:    {final_acc:.4f}")
    print(f"  F1 Macro:    {final_f1:.4f}")
    print(f"  Vocab size:  {len(vectorizer.vocabulary_)}")
    print(f"  Total time:  {elapsed:.2f}s")
    print(f"  Output dir:  {TFIDF_MODEL_DIR}")
    print("=" * 60)

    return metadata


if __name__ == "__main__":
    main()
