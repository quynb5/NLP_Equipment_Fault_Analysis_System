# 📖 Instructions — Hướng dẫn đọc tài liệu Project

## Tổng quan

Project này có **5 file tài liệu `.md`**, mỗi file phục vụ một mục đích khác nhau. Dưới đây là hướng dẫn mỗi file chứa gì, nên đọc khi nào, và cần hiểu những gì.

---

## 1. README.md

**Mục đích**: Hướng dẫn cài đặt và chạy project

**Nội dung chính**:
- Giới thiệu ngắn gọn project
- Hướng dẫn cài đặt môi trường (Python, dependencies)
- Cách chạy server, train model, chạy evaluation
- API endpoints cơ bản

**Khi nào đọc**: Khi mới clone project, cần setup và chạy lần đầu

**Cần hiểu**:
- [ ] Cách cài đặt environment và dependencies
- [ ] Cách start server (`python main.py`)
- [ ] Các API endpoints có sẵn

---

## 2. PROJECT_OVERVIEW.md

**Mục đích**: Tài liệu kỹ thuật tổng quan toàn bộ hệ thống

**Nội dung chính**:
- Kiến trúc Multi-Engine (PhoBERT + TF-IDF)
- Cấu trúc thư mục chi tiết
- NLP Pipeline 6 bước (Tiền xử lý → Tokenization → Keyword → Classification → Severity → Recommendation)
- Dual-mode Classification (Fine-tuned vs Zero-shot)
- Bảng 10 loại lỗi và severity
- API endpoints, stack công nghệ
- Sequence diagram luồng xử lý

**Khi nào đọc**: Khi cần hiểu tổng thể hệ thống hoạt động như thế nào

**Cần hiểu**:
- [ ] Kiến trúc Multi-Engine và Strategy Pattern (BaseNLPEngine → PhoBERTEngine, TFIDFEngine)
- [ ] NLP Pipeline 6 bước — đặc biệt bước 4 (Dual-mode Classification)
- [ ] Sự khác biệt giữa PhoBERT Engine và TF-IDF Engine
- [ ] 10 loại lỗi thiết bị và cách tính severity score
- [ ] Luồng xử lý từ user input → API → Engine → Database → Response

---

## 3. FinetuningModelPhoBert.md

**Mục đích**: Giải thích kỹ thuật Fine-tuning PhoBERT — phần **học thuật quan trọng nhất**

**Nội dung chính**:
- Transfer Learning là gì và tại sao dùng
- Quy trình fine-tuning 4 bước (Pre-trained → Add Head → Train → Inference)
- Kiến trúc PhoBERTClassifier (PhoBERT + Dropout + Linear)
- 6 ưu điểm của Transfer Learning
- So sánh với các phương pháp khác
- Kết quả trước/sau fine-tuning
- Thuật ngữ học thuật
- Tài liệu tham khảo (papers)

**Khi nào đọc**: Khi cần báo cáo, thuyết trình về phương pháp kỹ thuật

**Cần hiểu**:
- [ ] **Transfer Learning** là gì — tại sao không train từ đầu
- [ ] **Pre-training vs Fine-tuning** — 2 phase riêng biệt
- [ ] **Kiến trúc model**: PhoBERT (135M params) + Linear head (7,690 params)
- [ ] **[CLS] token pooling** — cách lấy embedding đại diện cho cả câu
- [ ] **Masked Language Modeling (MLM)** — bài toán pre-training
- [ ] **Contextual embedding** — tại sao "nóng" có embedding khác nhau tùy context
- [ ] **Ưu điểm**: Data efficiency, semantic understanding, generalization
- [ ] **Hyperparameters**: AdamW, warmup, cosine decay, early stopping, dropout
- [ ] Kết quả: 73.83% (zero-shot) → **89.93%** (fine-tuned) = +16.1%

---

## 4. BENCHMARK.md

**Mục đích**: So sánh chi tiết PhoBERT vs TF-IDF — bằng chứng thực nghiệm

**Nội dung chính**:
- Bảng so sánh tổng thể (accuracy, F1, precision, recall, latency)
- Per-class F1-score comparison
- **8 mẫu cụ thể** PhoBERT đúng, TF-IDF sai — phân tích chi tiết tại sao
- **7 mẫu cụ thể** TF-IDF đúng, PhoBERT sai — phân tích điểm yếu
- **5 Pattern** giải thích TF-IDF kém hơn
- Trade-offs giữa 2 engines
- **5 lý do thuyết phục** nên chọn PhoBERT thay vì TF-IDF

**Khi nào đọc**: Khi cần chứng minh PhoBERT tốt hơn TF-IDF cho báo cáo

**Cần hiểu**:
- [ ] PhoBERT (89.93%) vs TF-IDF (89.26%) — PhoBERT thắng overall
- [ ] **5 Pattern TF-IDF kém**:
  1. Root Cause vs Symptom Confusion — TF-IDF bắt triệu chứng phụ, bỏ qua root cause
  2. Multi-symptom Keyword Dominance — TF-IDF bị keyword mạnh nhất chi phối
  3. Paraphrasing — TF-IDF fail khi dùng cách diễn đạt khác
  4. Fine-grained vs Coarse Classification — TF-IDF default class chung chung
  5. Confidence Gap — PhoBERT tự tin hơn (0.87 vs 0.26)
- [ ] **5 lý do chọn PhoBERT** (dù accuracy gần nhau):
  1. Tiềm năng scale — thêm data PhoBERT cải thiện mạnh, TF-IDF saturate
  2. Semantic understanding — xử lý real-world input đa dạng
  3. Root cause detection — phân loại đúng nguyên nhân gốc
  4. Confidence calibration — tin cậy hơn khi deploy production
  5. Multi-task extensibility — mở rộng cho NER, QA, sentiment...
- [ ] **PhoBERT yếu ở đâu**: Nhận diện "Hoạt động ổn định" (F1=0.93 vs TF-IDF 1.00)
- [ ] **Trade-offs**: PhoBERT chậm hơn 10x (30ms vs 3ms), nặng hơn 10,000x (540MB vs 50KB)

---

## 5. EVALUATION_REPORT.md

**Mục đích**: Báo cáo kết quả evaluation chính thức

**Nội dung chính**:
- Kết quả evaluation trên bộ 149 mẫu test
- Confusion matrix
- Per-class metrics
- Danh sách mẫu bị misclassified

**Khi nào đọc**: Khi cần số liệu chính xác để trích dẫn

**Cần hiểu**:
- [ ] Cách đọc confusion matrix
- [ ] Ý nghĩa các metrics: Accuracy, Precision, Recall, F1-score
- [ ] Macro average vs Weighted average

---

## Thứ tự đọc được khuyến nghị

```
1. README.md              ← Setup & chạy project
2. PROJECT_OVERVIEW.md    ← Hiểu tổng thể hệ thống
3. FinetuningModelPhoBert.md  ← Hiểu kỹ thuật fine-tuning (học thuật)
4. BENCHMARK.md           ← Bằng chứng PhoBERT > TF-IDF
5. EVALUATION_REPORT.md   ← Số liệu chính xác
```

---

## Tóm tắt nhanh — Những điểm quan trọng nhất cần nắm

| Câu hỏi | Trả lời |
|---|---|
| **Hệ thống làm gì?** | Phân loại lỗi thiết bị CN từ mô tả tiếng Việt (10 classes) |
| **Dùng kỹ thuật gì?** | Transfer Learning — Fine-tuning PhoBERT |
| **PhoBERT là gì?** | Pre-trained Language Model cho tiếng Việt (VinAI, 135M params) |
| **Fine-tuning là gì?** | Tinh chỉnh model pre-trained cho task cụ thể bằng dữ liệu ít |
| **Accuracy bao nhiêu?** | PhoBERT: 89.93%, TF-IDF: 89.26% |
| **Tại sao PhoBERT tốt hơn?** | Hiểu ngữ nghĩa, context, paraphrase — không chỉ match keyword |
| **Tại sao TF-IDF vẫn cần?** | Nhanh 10x, nhẹ 10,000x, tốt cho baseline comparison |
| **Dữ liệu cần bao nhiêu?** | Chỉ 1,197 mẫu (nhờ transfer learning) |
