# 🔬 Fine-tuning PhoBERT cho Text Classification

## 1. Kỹ thuật sử dụng

Kỹ thuật này gọi là **Transfer Learning** (Học chuyển giao), cụ thể là **Fine-tuning a Pre-trained Language Model** cho bài toán **downstream task** (Text Classification).

> **"Fine-tuning BERT-based models for sequence classification"**
> (Devlin et al., 2019 — paper gốc BERT)

### Transfer Learning là gì?

Transfer Learning là phương pháp **tận dụng kiến thức** đã học từ một task/domain lớn (pre-training) để áp dụng cho một task/domain nhỏ hơn, cụ thể hơn (downstream task). Thay vì train model từ đầu (from scratch) với random weights, ta bắt đầu từ model đã có sẵn kiến thức ngôn ngữ → chỉ cần **tinh chỉnh (fine-tune)** cho bài toán cụ thể.

```
┌─────────────────────────────────────────────────────────┐
│            Transfer Learning Pipeline                    │
│                                                         │
│  Phase 1: Pre-training (đã làm sẵn bởi VinAI)          │
│  ┌─────────────────────────────────────────────┐        │
│  │ Dữ liệu: 20GB text tiếng Việt              │        │
│  │ Task: Masked Language Modeling (MLM)         │        │
│  │ Output: PhoBERT-base (135M parameters)       │        │
│  │ → Kiến thức ngôn ngữ tiếng Việt phong phú   │        │
│  └─────────────────────────────────────────────┘        │
│                      ↓                                  │
│  Phase 2: Fine-tuning (do chúng ta thực hiện)           │
│  ┌─────────────────────────────────────────────┐        │
│  │ Dữ liệu: 1,197 mẫu lỗi thiết bị có nhãn   │        │
│  │ Task: Fault Classification (10 classes)      │        │
│  │ Thêm: Linear classification head             │        │
│  │ → Model chuyên biệt phân loại lỗi thiết bị  │        │
│  └─────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Quy trình Fine-tuning đã thực hiện

### Bước 1 — Sử dụng Pre-trained PhoBERT

PhoBERT (`vinai/phobert-base`) là model Transformer đã được **pre-train** trên **20GB dữ liệu tiếng Việt** (báo, wiki, forum...) bằng bài toán **Masked Language Modeling (MLM)** — che một số từ trong câu rồi train model đoán lại:

```
Input:  "Motor bị [MASK] bất thường khi vận hành"
Output: "Motor bị  rung  bất thường khi vận hành"
```

Sau pre-training, PhoBERT đã có **kiến thức ngôn ngữ tiếng Việt**:
- Hiểu ngữ pháp, cú pháp tiếng Việt
- Hiểu quan hệ ngữ nghĩa giữa các từ
- Encode bất kỳ câu tiếng Việt nào thành **vector 768 chiều** chứa thông tin ngữ nghĩa

### Bước 2 — Thêm Classification Head

Thêm **1 layer Linear** lên trên PhoBERT để biến nó thành classifier:

```
┌──────────────────────────────────────────────────┐
│ PhoBERTClassifier (nn.Module)                    │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │ PhoBERT Encoder (vinai/phobert-base)       │  │
│  │   12 Transformer layers                    │  │
│  │   Hidden size: 768                         │  │
│  │   Attention heads: 12                      │  │
│  │   Parameters: ~135M (pre-trained)          │  │
│  │   ❄️ Embedding layer: FROZEN              │  │
│  └──────────────┬─────────────────────────────┘  │
│                 │ [CLS] token (768-dim)          │
│  ┌──────────────▼─────────────────────────────┐  │
│  │ Dropout(p=0.3) — Regularization            │  │
│  └──────────────┬─────────────────────────────┘  │
│                 │                                │
│  ┌──────────────▼─────────────────────────────┐  │
│  │ Linear(768, 10) — Classification Head      │  │
│  │   Parameters: 768 × 10 + 10 = 7,690       │  │
│  └──────────────┬─────────────────────────────┘  │
│                 │ logits (10-dim)                 │
│  ┌──────────────▼─────────────────────────────┐  │
│  │ Softmax → Probabilities (10 classes)       │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

### Bước 3 — Train trên dữ liệu có nhãn

| Thành phần | Chi tiết |
|---|---|
| **Dữ liệu** | 1,197 mẫu (957 train / 240 val) |
| **Optimizer** | AdamW (Adam with decoupled weight decay) |
| **Learning Rate** | 2e-5 (standard cho fine-tuning Transformer) |
| **Scheduler** | Linear warmup (10% steps) + Cosine decay |
| **Early Stopping** | Patience=5 (dừng nếu 5 epochs không cải thiện) |
| **Freeze** | Embedding layer frozen (tiết kiệm memory) |
| **Best Epoch** | 4/20 (val accuracy = 100%) |
| **Training Time** | ~150 giây (GPU) |

### Bước 4 — Inference (Sử dụng)

```
Input: "Motor nóng bất thường, rung mạnh"
  → PhoBERT Tokenize → [CLS] motor nóng bất_thường , rung mạnh [SEP]
  → PhoBERT Encode   → [CLS] embedding (768-dim)
  → Dropout           → Regularization
  → Linear(768, 10)  → 10 logits
  → Softmax           → [0.01, 0.01, 0.01, 0.01, 0.01, 0.92, 0.01, 0.01, 0.01, 0.01]
  → Output: "Quá nhiệt" (confidence=0.92)
```

---

## 3. Ưu điểm của Transfer Learning / Fine-tuning

### 3.1 Tiết kiệm dữ liệu (Data Efficiency)

| Phương pháp | Dữ liệu cần |
|---|:---:|
| Train from scratch | 10,000 - 100,000+ mẫu |
| **Fine-tuning PhoBERT** | **~1,200 mẫu** |

Pre-training trên 20GB text → model đã có kiến thức ngôn ngữ. Chỉ cần ít dữ liệu labeled để "dạy" thêm bài toán cụ thể. Đây là ưu điểm **quan trọng nhất** trong thực tế vì thu thập dữ liệu labeled rất tốn kém.

### 3.2 Hiểu ngữ nghĩa sâu (Semantic Understanding)

```
TF-IDF:    "vết ẩm ướt lan rộng"  →  không match keyword "rò rỉ"  →  ❌ Sai
PhoBERT:   "vết ẩm ướt lan rộng"  →  embedding gần "rò rỉ"       →  ✅ Đúng
```

- PhoBERT hiểu **paraphrase**: "vết ẩm ướt" ≈ "rò rỉ"
- Hiểu **world knowledge**: "bi vỡ mẻ" → liên quan đến bạc đạn
- TF-IDF chỉ match **exact keyword**, không hiểu ngữ nghĩa

### 3.3 Contextual Representation (Biểu diễn phụ thuộc ngữ cảnh)

Mỗi từ có embedding **khác nhau** tùy vào context xung quanh (nhờ Transformer self-attention):

```
"Motor nóng bất thường"    → embedding("nóng") = vector hướng fault
"Motor không nóng"         → embedding("nóng") = vector hướng normal (bị negation)
"Không nóng không ồn"      → embedding tổng thể hướng "hoạt động bình thường"
```

TF-IDF: từ "nóng" **luôn có cùng 1 weight** bất kể context → không phân biệt được phủ định.

### 3.4 Phân biệt Root Cause vs Symptom

```
Input: "Ốc siết bích nối bị gãy do rung, bích hở gây rò rỉ nhẹ"
         ^^^^^^^^^^^^^^^^^^^^^^                   ^^^^^^^^
         Root cause (cơ khí)                      Symptom (rò rỉ)

PhoBERT attention nhìn toàn câu → hiểu root cause = cơ khí  → ✅
TF-IDF bắt keyword "rò rỉ" (weight cao)                     → ❌
```

### 3.5 Generalization tốt

Fine-tuned model hoạt động tốt với câu mô tả **chưa từng thấy** trong training data. Nhờ pre-trained knowledge:
- "Motor kéo không nổi" (chưa thấy) → hiểu ≈ "quá tải" (đã thấy)
- "Tiếng lạch cạch từ bên trong" (chưa thấy) → hiểu ≈ "âm thanh bất thường" (đã thấy)

### 3.6 Hội tụ nhanh (Fast Convergence)

```
Epoch 1: Val Acc = 85.0%     ← Đã khá tốt ngay epoch đầu
Epoch 2: Val Acc = 94.6%
Epoch 3: Val Acc = 98.3%
Epoch 4: Val Acc = 100.0%    ← Best (chỉ 4 epochs!)
```

Nhờ pre-trained weights là **starting point tốt**, không cần train từ random initialization (thường cần 50-100+ epochs).

---

## 4. So sánh với các phương pháp khác

| Phương pháp | Data cần | Semantic | Accuracy | Tốc độ | Kích thước |
|---|:---:|:---:|:---:|:---:|:---:|
| **Rule-based** (keyword) | 0 | ❌ | ~60% | Rất nhanh | ~0 |
| **TF-IDF + LR** | ~1,000+ | ❌ | 89.26% | Nhanh (~3ms) | ~50KB |
| **PhoBERT Zero-shot** | 0 | ✅ Một phần | 73.83% | Chậm (~30ms) | 540MB |
| **PhoBERT Fine-tuned** ✅ | ~1,200 | ✅ Đầy đủ | **89.93%** | Chậm (~30ms) | 540MB |
| **Train from scratch** | 10,000+ | ✅ | Có thể cao hơn | Rất chậm | Tùy |

---

## 5. Kết quả đạt được

| Metric | Trước Fine-tuning | Sau Fine-tuning | Cải thiện |
|---|:---:|:---:|:---:|
| **Accuracy** | 73.83% (zero-shot) | **89.93%** | +16.1% |
| **F1 (macro)** | 74.02% | **89.99%** | +16.0% |
| **Misclassified** | 39/149 | **15/149** | -24 mẫu |

So với TF-IDF baseline:

| Metric | PhoBERT (Fine-tuned) | TF-IDF |
|---|:---:|:---:|
| **Accuracy** | **89.93%** ✅ | 89.26% |
| **F1 (macro)** | **89.99%** | 89.31% |

---

## 6. Thuật ngữ học thuật

| Thuật ngữ | Giải thích |
|---|---|
| **Transfer Learning** | Học chuyển giao — dùng knowledge từ task/domain khác |
| **Fine-tuning** | Tinh chỉnh model pre-trained cho task cụ thể |
| **Pre-trained Language Model (PLM)** | Model ngôn ngữ đã pre-train (PhoBERT, BERT, GPT...) |
| **Downstream Task** | Bài toán cụ thể cần giải (text classification) |
| **[CLS] Token Pooling** | Dùng embedding token [CLS] đại diện cho cả câu |
| **Masked Language Modeling (MLM)** | Bài toán pre-training: đoán từ bị che |
| **Transformer / Self-Attention** | Kiến trúc mạng xử lý song song, tính attention giữa mọi cặp từ |
| **AdamW Optimizer** | Adam with decoupled weight decay — optimizer chuẩn cho Transformers |
| **Learning Rate Warmup** | Tăng dần LR ở đầu training tránh gradient explosion |
| **Cosine Decay** | Giảm LR theo hàm cosine, mượt hơn step decay |
| **Early Stopping** | Dừng training sớm khi validation metric không cải thiện |
| **Dropout** | Kỹ thuật regularization — random tắt neurons khi training |

---

## 7. Các file liên quan

| File | Mô tả |
|---|---|
| `backend/training/train_phobert.py` | Pipeline fine-tuning hoàn chỉnh |
| `backend/training/config.py` | Hyperparameters |
| `backend/training/data_preparation.py` | Data loading & augmentation |
| `backend/core/phobert_engine.py` | Engine sử dụng fine-tuned model |
| `backend/resources/phobert-finetuned/` | Model artifacts đã train |

---

## 8. Tham khảo

1. **PhoBERT** — Nguyen & Nguyen (2020): *"PhoBERT: Pre-trained language models for Vietnamese"* — VinAI Research
2. **BERT** — Devlin et al. (2019): *"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"* — Google AI
3. **Transfer Learning Survey** — Ruder (2019): *"Neural Transfer Learning for NLP"*
4. **AdamW** — Loshchilov & Hutter (2019): *"Decoupled Weight Decay Regularization"*
