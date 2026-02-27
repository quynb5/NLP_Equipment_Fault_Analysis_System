# 🔧 NLP Equipment Fault Analysis System

Hệ thống phân tích lỗi thiết bị công nghiệp bằng xử lý ngôn ngữ tự nhiên (NLP), hỗ trợ 2 engine: **PhoBERT** (zero-shot semantic similarity) và **TF-IDF** (Logistic Regression classifier).

---

## 📁 Cấu trúc dự án

```
nlp/
├── main.py                          # Entry point — chạy server
├── requirements.txt                 # Thư viện cần cài
├── backend/
│   ├── app.py                       # FastAPI routes (/analyze, /history, ...)
│   ├── core/
│   │   ├── base_engine.py           # Abstract base class cho engine
│   │   ├── phobert_engine.py        # PhoBERT engine (semantic similarity)
│   │   ├── tfidf_engine.py          # TF-IDF engine (classifier)
│   │   └── engine_factory.py        # Factory pattern chọn engine
│   ├── model/
│   │   └── schemas.py               # Pydantic schemas (request/response)
│   ├── database/
│   │   └── database.py              # SQLite lưu lịch sử phân tích
│   ├── training/
│   │   ├── data_preparation.py      # Tạo training data từ FAULT_REFERENCES
│   │   ├── train_tfidf.py           # Training pipeline TF-IDF
│   │   └── config.py                # Cấu hình training
│   ├── evaluation/
│   │   ├── evaluation_dataset.json  # Dataset đánh giá (149 mẫu)
│   │   ├── run_evaluation.py        # Chạy đánh giá engine
│   │   ├── evaluator.py             # Predict helper
│   │   ├── metrics.py               # Tính metrics + confusion matrix
│   │   ├── latency.py               # Đo latency
│   │   └── results/                 # Kết quả evaluation (auto-generated)
│   ├── test/
│   │   ├── test_engine.py           # Unit tests cho engine
│   │   └── test_phase2.py           # Integration tests
│   └── resources/
│       └── tfidf/                   # Model TF-IDF đã train (auto-generated)
└── frontend/
    └── templates/
        └── ui.html                  # Giao diện web
```

---

## 🚀 Hướng dẫn cài đặt & chạy

### Bước 1: Tạo môi trường Conda

```bash
# Tạo environment mới với Python 3.10
conda create -n mlops python=3.10 -y

# Kích hoạt environment
conda activate mlops
```

### Bước 2: Cài đặt thư viện

```bash
# Di chuyển vào thư mục dự án
cd /mnt/atin/QuyNB/project/master_project/nlp

# Cài đặt tất cả thư viện
pip install -r requirements.txt
```

> **Lưu ý:** Lần đầu chạy, PhoBERT model (~1.3GB) sẽ tự động tải từ HuggingFace.

---

### Bước 3: Training TF-IDF Model

```bash
# Train TF-IDF classifier từ FAULT_REFERENCES data
python -m backend.training.train_tfidf
```

**Output:** Các file model sẽ được lưu tại `backend/resources/tfidf/`:
- `vectorizer.pkl` — TF-IDF vectorizer
- `classifier.pkl` — Logistic Regression classifier
- `label_encoder.pkl` — Label encoder
- `metadata.json` — Metadata & version info

> ⚠️ **Bắt buộc chạy bước này trước khi dùng TF-IDF engine.** PhoBERT engine không cần training.

---

### Bước 4: Chạy server (Production)

```bash
# Chạy FastAPI server
python main.py
```

Server sẽ chạy tại: **http://localhost:10805**

| Endpoint | Method | Mô tả |
|---|---|---|
| `/` | GET | Giao diện web |
| `/analyze` | POST | Phân tích lỗi thiết bị |
| `/history` | GET | Lịch sử phân tích |
| `/history/{id}` | GET | Chi tiết 1 bản ghi |
| `/docs` | GET | Swagger API docs |

**Ví dụ gọi API:**

```bash
curl -X POST http://localhost:10805/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "equipment": "Motor bơm nước",
    "description": "Motor rung mạnh kèm tiếng kim loại va chạm",
    "engine": "phobert"
  }'
```

Tham số `engine` có thể là `"phobert"` hoặc `"tfidf"`.

---

### Bước 5: Đánh giá model (Evaluation)

```bash
# Đánh giá cả 2 engine
python -m backend.evaluation.run_evaluation --engine all

# Đánh giá riêng từng engine
python -m backend.evaluation.run_evaluation --engine phobert
python -m backend.evaluation.run_evaluation --engine tfidf
```

**Output:** Kết quả được lưu tại `backend/evaluation/results/`:
- `confusion_matrix_*.png` — Ma trận nhầm lẫn
- `evaluation_report_*.json` — Báo cáo chi tiết (JSON)
- `evaluation_report_*.txt` — Báo cáo chi tiết (text)
- `comparison_report.json` — So sánh 2 engine

---

### Bước 6: Chạy tests

```bash
# Unit tests (không cần server đang chạy)
python -m backend.test.test_engine

# End-to-end tests (cần server đang chạy ở bước 4)
python test_e2e.py
```

---

## 📊 Kết quả đánh giá hiện tại

Dataset: **149 mẫu** (unseen data, 10 fault types)

| Metric | TF-IDF | PhoBERT |
|---|---|---|
| Accuracy | **89.26%** | 73.83% |
| F1 Macro | **0.8931** | 0.7402 |
| Latency | **2.9ms** | 31.0ms |

---

## 🏷️ 10 Loại lỗi hỗ trợ

| # | Loại lỗi | Mô tả |
|---|---|---|
| 1 | Hoạt động ổn định | Thiết bị bình thường |
| 2 | Quá nhiệt | Nhiệt độ cao bất thường |
| 3 | Hỏng bạc đạn / vòng bi | Rung + tiếng kim loại |
| 4 | Cháy cuộn dây / cháy motor | Mùi cháy + bốc khói |
| 5 | Sự cố điện | Chập, rò, quá tải điện |
| 6 | Quá tải cơ khí | Nóng + rung do quá tải |
| 7 | Rò rỉ hệ thống | Rò dầu, nước, khí |
| 8 | Hư hỏng cơ khí | Nứt, gãy, mòn, gỉ sét |
| 9 | Âm thanh bất thường | Tiếng ồn, kêu lạ |
| 10 | Giảm hiệu suất | Chạy chậm, yếu, kém |
