"""
TASK 06 — Benchmark Runner Script
===================================
Chạy toàn bộ evaluation pipeline:
  1. Load dataset
  2. Predict toàn bộ
  3. Tính confusion matrix
  4. Tính classification metrics
  5. Tính latency
  6. Xuất report (CSV, PNG, JSON, TXT)

Usage:
  cd /mnt/atin/QuyNB/project/master_project/nlp
  python -m backend.evaluation.run_evaluation
"""
import json
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.evaluation.evaluator import predict_label
from backend.evaluation.metrics import (
    compute_confusion_matrix,
    save_confusion_matrix_csv,
    save_confusion_matrix_heatmap,
    compute_classification_metrics,
    save_metrics_json,
    save_metrics_txt,
)
from backend.evaluation.latency import measure_latency, print_latency_stats


# 10 fault type labels (thứ tự hiển thị)
LABELS = [
    "Hoạt động ổn định",
    "Quá nhiệt",
    "Hỏng bạc đạn / vòng bi",
    "Cháy cuộn dây / cháy motor",
    "Sự cố điện",
    "Quá tải cơ khí",
    "Rò rỉ hệ thống",
    "Hư hỏng cơ khí",
    "Âm thanh bất thường",
    "Giảm hiệu suất",
]


def load_dataset(path: str = None) -> list[dict]:
    """Load test dataset từ JSON file."""
    if path is None:
        path = str(Path(__file__).resolve().parent / "test_dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def run_evaluation():
    """Chạy toàn bộ evaluation pipeline."""
    print("=" * 60)
    print("  🚀 MODEL EVALUATION — PhoBERT NLP Engine")
    print("=" * 60)

    # 1. Load dataset
    print("\n📂 Loading test dataset...")
    dataset = load_dataset()
    print(f"   Loaded {len(dataset)} samples")

    texts = [d["description"] for d in dataset]
    y_true = [d["true_label"] for d in dataset]

    # 2. Predict toàn bộ
    print("\n🔮 Running predictions...")
    y_pred = []
    for i, text in enumerate(texts):
        pred = predict_label(text)
        y_pred.append(pred)
        if (i + 1) % 25 == 0 or (i + 1) == len(texts):
            print(f"   [{i+1}/{len(texts)}] done")

    # 3. Confusion Matrix
    print("\n📊 Computing confusion matrix...")
    cm_df = compute_confusion_matrix(y_true, y_pred, labels=LABELS)
    save_confusion_matrix_csv(cm_df)
    save_confusion_matrix_heatmap(cm_df)

    # Print confusion matrix to console
    print("\nConfusion Matrix:")
    print(cm_df.to_string())

    # 4. Classification Metrics
    print("\n📈 Computing classification metrics...")
    metrics = compute_classification_metrics(y_true, y_pred, labels=LABELS)

    print(f"\n   Accuracy:        {metrics['accuracy']:.4f}")
    print(f"   Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"   Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"   F1-score (macro):  {metrics['f1_macro']:.4f}")

    print("\n   Per-class metrics:")
    for label, m in metrics["per_class"].items():
        print(f"     {label:35s}  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1-score']:.2f}  (n={m['support']})")

    # 5. Latency
    print("\n⏱️  Measuring latency...")
    latency_stats = measure_latency(texts)
    print_latency_stats(latency_stats)

    # Add latency to metrics
    metrics["latency"] = {
        "mean_ms": latency_stats["mean_ms"],
        "min_ms": latency_stats["min_ms"],
        "max_ms": latency_stats["max_ms"],
        "p95_ms": latency_stats["p95_ms"],
    }

    # 6. Export reports
    print("\n💾 Saving reports...")
    save_metrics_json(metrics)
    save_metrics_txt(y_true, y_pred, labels=LABELS, latency_stats=latency_stats)

    # Print misclassified samples
    misclassified = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            misclassified.append({
                "index": i,
                "text": texts[i][:60],
                "true": true,
                "predicted": pred,
            })

    if misclassified:
        print(f"\n⚠️  Misclassified samples: {len(misclassified)}/{len(y_true)}")
        for m in misclassified[:20]:  # Show max 20
            print(f"   [{m['index']}] \"{m['text']}...\"")
            print(f"        True: {m['true']}  →  Predicted: {m['predicted']}")
    else:
        print("\n✅ No misclassified samples!")

    print("\n" + "=" * 60)
    print(f"  ✅ EVALUATION COMPLETE — {len(dataset)} samples")
    print(f"     Accuracy: {metrics['accuracy']:.2%}")
    print(f"     F1 (macro): {metrics['f1_macro']:.2%}")
    print(f"     Avg latency: {latency_stats['mean_ms']:.1f} ms")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    run_evaluation()
