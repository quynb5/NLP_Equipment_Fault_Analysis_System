"""
TASK 06 — Benchmark Runner Script (Multi-Engine)
===================================================
Chạy evaluation pipeline cho 1 hoặc nhiều engine:
  1. Load dataset
  2. Predict toàn bộ (per engine)
  3. Tính confusion matrix
  4. Tính classification metrics
  5. Tính latency
  6. Xuất report (CSV, PNG, JSON, TXT)
  7. So sánh engines (nếu chạy multi-engine)

Usage:
  cd /mnt/atin/QuyNB/project/master_project/nlp

  # Evaluate PhoBERT only
  python -m backend.evaluation.run_evaluation --engine phobert

  # Evaluate TF-IDF only
  python -m backend.evaluation.run_evaluation --engine tfidf

  # Evaluate & compare both engines
  python -m backend.evaluation.run_evaluation --engine all
"""
import argparse
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
    RESULTS_DIR,
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
        path = str(Path(__file__).resolve().parent / "evaluation_dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def evaluate_engine(engine_name: str, texts: list[str], y_true: list[str]) -> dict:
    """
    Chạy evaluation pipeline cho 1 engine.

    Returns:
        dict chứa metrics, latency, confusion matrix, misclassified items
    """
    engine_upper = engine_name.upper()
    print(f"\n{'=' * 60}")
    print(f"  🚀 EVALUATION — {engine_upper} Engine")
    print(f"{'=' * 60}")

    # Predict
    print(f"\n🔮 [{engine_upper}] Running predictions...")
    y_pred = []
    for i, text in enumerate(texts):
        pred = predict_label(text, engine_name=engine_name)
        y_pred.append(pred)
        if (i + 1) % 25 == 0 or (i + 1) == len(texts):
            print(f"   [{i+1}/{len(texts)}] done")

    # Confusion Matrix
    print(f"\n📊 [{engine_upper}] Computing confusion matrix...")
    cm_df = compute_confusion_matrix(y_true, y_pred, labels=LABELS)
    save_confusion_matrix_csv(
        cm_df,
        filepath=str(RESULTS_DIR / f"confusion_matrix_{engine_name}.csv"),
    )
    save_confusion_matrix_heatmap(
        cm_df,
        filepath=str(RESULTS_DIR / f"confusion_matrix_{engine_name}.png"),
        title=f"Confusion Matrix — {engine_upper} Engine (10 Fault Types)",
    )

    # Classification Metrics
    print(f"\n📈 [{engine_upper}] Computing classification metrics...")
    metrics = compute_classification_metrics(y_true, y_pred, labels=LABELS)
    metrics["engine"] = engine_name

    print(f"\n   Accuracy:          {metrics['accuracy']:.4f}")
    print(f"   Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"   Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"   F1-score (macro):  {metrics['f1_macro']:.4f}")

    print(f"\n   Per-class metrics:")
    for label, m in metrics["per_class"].items():
        print(f"     {label:35s}  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1-score']:.2f}  (n={m['support']})")

    # Latency
    print(f"\n⏱️  [{engine_upper}] Measuring latency...")
    latency_stats = measure_latency(texts, engine_name=engine_name)
    print_latency_stats(latency_stats)

    metrics["latency"] = {
        "mean_ms": latency_stats["mean_ms"],
        "min_ms": latency_stats["min_ms"],
        "max_ms": latency_stats["max_ms"],
        "p95_ms": latency_stats["p95_ms"],
    }

    # Save per-engine reports
    save_metrics_json(
        metrics,
        filepath=str(RESULTS_DIR / f"evaluation_report_{engine_name}.json"),
    )
    save_metrics_txt(
        y_true, y_pred, labels=LABELS,
        latency_stats=latency_stats,
        filepath=str(RESULTS_DIR / f"evaluation_report_{engine_name}.txt"),
        engine_name=engine_name,
    )

    # Misclassified
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
        print(f"\n⚠️  [{engine_upper}] Misclassified: {len(misclassified)}/{len(y_true)}")
        for m in misclassified[:10]:
            print(f'   [{m["index"]}] "{m["text"]}..."')
            print(f"        True: {m['true']}  →  Predicted: {m['predicted']}")
    else:
        print(f"\n✅ [{engine_upper}] No misclassified samples!")

    print(f"\n{'=' * 60}")
    print(f"  ✅ {engine_upper} EVALUATION COMPLETE — {len(texts)} samples")
    print(f"     Accuracy: {metrics['accuracy']:.2%}")
    print(f"     F1 (macro): {metrics['f1_macro']:.2%}")
    print(f"     Avg latency: {latency_stats['mean_ms']:.1f} ms")
    print(f"{'=' * 60}")

    return {
        "metrics": metrics,
        "latency": latency_stats,
        "y_pred": y_pred,
        "misclassified": misclassified,
    }


def print_comparison(results: dict[str, dict]):
    """In bảng so sánh giữa các engine."""
    print(f"\n{'=' * 70}")
    print("  🔀 ENGINE COMPARISON")
    print(f"{'=' * 70}")

    header = f"{'Metric':<25}"
    for eng in results:
        header += f"{eng.upper():>20}"
    print(header)
    print("-" * 70)

    # Metrics rows
    metric_keys = [
        ("Accuracy", "accuracy"),
        ("Precision (macro)", "precision_macro"),
        ("Recall (macro)", "recall_macro"),
        ("F1-score (macro)", "f1_macro"),
    ]
    for display, key in metric_keys:
        row = f"  {display:<23}"
        for eng in results:
            val = results[eng]["metrics"][key]
            row += f"{val:>20.4f}"
        print(row)

    # Latency rows
    print("-" * 70)
    latency_keys = [
        ("Latency Mean (ms)", "mean_ms"),
        ("Latency P95 (ms)", "p95_ms"),
        ("Latency Min (ms)", "min_ms"),
        ("Latency Max (ms)", "max_ms"),
    ]
    for display, key in latency_keys:
        row = f"  {display:<23}"
        for eng in results:
            val = results[eng]["latency"][key]
            row += f"{val:>20.2f}"
        print(row)

    # Misclassified
    print("-" * 70)
    row = f"  {'Misclassified':<23}"
    for eng in results:
        n = len(results[eng]["misclassified"])
        row += f"{n:>20}"
    print(row)

    print(f"{'=' * 70}")

    # Save comparison JSON
    comparison = {}
    for eng, data in results.items():
        comparison[eng] = {
            **data["metrics"],
            "latency": data["latency"],
            "misclassified_count": len(data["misclassified"]),
        }
        # Remove verbose fields
        comparison[eng].pop("per_class", None)

    comp_path = str(RESULTS_DIR / "comparison_report.json")
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"✅ Comparison report saved: {comp_path}")


def run_evaluation(engine_name: str = "all"):
    """
    Main evaluation entry point.

    Args:
        engine_name: "phobert", "tfidf", or "all" (compare both)
    """
    # Load dataset
    print("\n📂 Loading test dataset...")
    dataset = load_dataset()
    print(f"   Loaded {len(dataset)} samples")

    texts = [d["description"] for d in dataset]
    y_true = [d["true_label"] for d in dataset]

    engines = ["phobert", "tfidf"] if engine_name == "all" else [engine_name]
    results = {}

    for eng in engines:
        results[eng] = evaluate_engine(eng, texts, y_true)

    # Comparison
    if len(results) > 1:
        print_comparison(results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Engine Evaluation")
    parser.add_argument(
        "--engine", "-e",
        choices=["phobert", "tfidf", "all"],
        default="all",
        help="Engine to evaluate (default: all)",
    )
    args = parser.parse_args()
    run_evaluation(args.engine)
