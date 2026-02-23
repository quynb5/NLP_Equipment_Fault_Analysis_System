# ✅ EVALUATION REPORT — PhoBERT NLP Engine

## Kết quả thực thi

Toàn bộ 8 tasks đã triển khai thành công và chạy evaluation trên **150 mẫu test** (15 mẫu × 10 loại lỗi).

## Files đã tạo/sửa

| File | Task |
|---|---|
| `backend/evaluation/test_dataset.json` | 01 — 150 labeled samples |
| `backend/evaluation/evaluator.py` | 02 — Prediction wrapper |
| `backend/evaluation/metrics.py` | 03+04+07 — Metrics + Heatmap |
| `backend/evaluation/latency.py` | 05 — Latency measurement |
| `backend/evaluation/run_evaluation.py` | 06 — Runner script |
| `backend/app.py` | 08 — `GET /evaluation/summary` |

---

## Kết quả đánh giá

### Overall Metrics

| Metric | Score |
|---|---|
| **Accuracy** | **84.00%** |
| **Precision (macro)** | **89.88%** |
| **Recall (macro)** | **84.00%** |
| **F1-score (macro)** | **84.18%** |

### Per-class Metrics

| Loại lỗi | Precision | Recall | F1-Score |
|---|---|---|---|
| Cháy cuộn dây / cháy motor | 1.00 | 1.00 | **1.00** |
| Sự cố điện | 1.00 | 0.93 | **0.97** |
| Quá tải cơ khí | 0.83 | 1.00 | **0.91** |
| Quá nhiệt | 1.00 | 0.80 | **0.89** |
| Giảm hiệu suất | 0.92 | 0.80 | **0.86** |
| Hư hỏng cơ khí | 1.00 | 0.73 | **0.85** |
| Hỏng bạc đạn / vòng bi | 0.71 | 1.00 | **0.83** |
| Rò rỉ hệ thống | 1.00 | 0.67 | **0.80** |
| Hoạt động ổn định | 0.52 | 1.00 | **0.68** |
| Âm thanh bất thường | 1.00 | 0.47 | **0.64** |

### Confusion Matrix

> Xem heatmap: `EVALUATION_REPORT.png` hoặc `backend/evaluation/results/confusion_matrix.png`

**Nhầm lẫn chính:**
- **Âm thanh bất thường** → bị nhầm thành **Hỏng bạc đạn** (6/15) — vì cả hai đều liên quan đến tiếng ồn + rung
- **Rò rỉ hệ thống** → bị nhầm thành **Hoạt động ổn định** (5/15) — một số mô tả rò rỉ nhẹ bị phân loại là bình thường
- **Hư hỏng cơ khí** → bị nhầm thành **Hoạt động ổn định** (3/15) và **Quá tải cơ khí** (1/15)
- **Giảm hiệu suất** → bị nhầm thành **Hoạt động ổn định** (3/15)

### Latency (CPU)

| Metric | Value |
|---|---|
| Mean | **21.24 ms** |
| Min | 19.15 ms |
| Max | 29.45 ms |
| P95 | 22.78 ms |

---

## Output Files

- `backend/evaluation/results/confusion_matrix.png` — Heatmap 10×10
- `backend/evaluation/results/confusion_matrix.csv` — Ma trận CSV
- `backend/evaluation/results/evaluation_report.json` — Metrics JSON
- `backend/evaluation/results/evaluation_report.txt` — Full classification report

## Cách chạy lại

```bash
cd /mnt/atin/QuyNB/project/master_project/nlp
python -m backend.evaluation.run_evaluation
```
