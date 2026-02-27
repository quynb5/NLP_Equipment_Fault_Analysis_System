"""
TASK 03 + 04 + 07 — Confusion Matrix, Classification Metrics & Visualization
==============================================================================
- Confusion Matrix (numpy/pandas) → CSV + PNG heatmap
- Classification Report (Accuracy, Precision, Recall, F1) → JSON + TXT
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# Output directory
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TASK 03 — Confusion Matrix
# ============================================================

def compute_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> pd.DataFrame:
    """
    Tính confusion matrix và trả về dưới dạng pandas DataFrame.

    Args:
        y_true: Danh sách labels thực tế
        y_pred: Danh sách labels dự đoán
        labels: Danh sách tất cả labels (thứ tự hiển thị)

    Returns:
        pandas DataFrame — confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df = pd.DataFrame(cm, index=labels, columns=labels)
    return df


def save_confusion_matrix_csv(cm_df: pd.DataFrame, filepath: str = None):
    """Lưu confusion matrix thành CSV."""
    if filepath is None:
        filepath = str(RESULTS_DIR / "confusion_matrix.csv")
    cm_df.to_csv(filepath, encoding="utf-8-sig")
    print(f"✅ Confusion matrix CSV saved: {filepath}")
    return filepath


# ============================================================
# TASK 07 — Visualization (Heatmap)
# ============================================================

def save_confusion_matrix_heatmap(
    cm_df: pd.DataFrame,
    filepath: str = None,
    figsize: tuple = (14, 11),
    title: str = None,
):
    """
    Vẽ và lưu confusion matrix dạng heatmap.

    Args:
        cm_df: Confusion matrix DataFrame
        filepath: Đường dẫn file output (.png)
        figsize: Kích thước figure
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns

    if filepath is None:
        filepath = str(RESULTS_DIR / "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=figsize)

    # Shorten labels for display
    short_labels = []
    for label in cm_df.index:
        if len(label) > 20:
            short_labels.append(label[:18] + "…")
        else:
            short_labels.append(label)

    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short_labels,
        yticklabels=short_labels,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    ax.set_title(title or "Confusion Matrix — NLP Engine (10 Fault Types)", fontsize=14, pad=15)

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Confusion matrix heatmap saved: {filepath}")
    return filepath


# ============================================================
# TASK 04 — Classification Metrics
# ============================================================

def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict:
    """
    Tính các chỉ số phân loại.

    Returns:
        dict chứa accuracy, precision, recall, f1 (macro + per-class)
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    # Per-class report
    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "per_class": {
            label: {
                "precision": round(report_dict[label]["precision"], 4),
                "recall": round(report_dict[label]["recall"], 4),
                "f1-score": round(report_dict[label]["f1-score"], 4),
                "support": report_dict[label]["support"],
            }
            for label in labels
            if label in report_dict
        },
    }


def save_metrics_json(metrics: dict, filepath: str = None):
    """Lưu metrics thành JSON."""
    if filepath is None:
        filepath = str(RESULTS_DIR / "evaluation_report.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"✅ Evaluation report JSON saved: {filepath}")
    return filepath


def save_metrics_txt(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    latency_stats: dict = None,
    filepath: str = None,
    engine_name: str = "phobert",
):
    """Lưu classification report dạng text."""
    if filepath is None:
        filepath = str(RESULTS_DIR / "evaluation_report.txt")

    report_str = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"  EVALUATION REPORT — {engine_name.upper()} Engine\n")
        f.write("=" * 70 + "\n\n")
        f.write("Classification Report:\n\n")
        f.write(report_str)
        f.write("\n")

        if latency_stats:
            f.write("\n" + "=" * 70 + "\n")
            f.write("  LATENCY STATS (CPU)\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"  Mean:  {latency_stats['mean_ms']:.2f} ms\n")
            f.write(f"  Min:   {latency_stats['min_ms']:.2f} ms\n")
            f.write(f"  Max:   {latency_stats['max_ms']:.2f} ms\n")
            f.write(f"  P95:   {latency_stats['p95_ms']:.2f} ms\n")
            f.write(f"  Total: {latency_stats['total_samples']} samples\n")
            f.write("\n")

    print(f"✅ Evaluation report TXT saved: {filepath}")
    return filepath
