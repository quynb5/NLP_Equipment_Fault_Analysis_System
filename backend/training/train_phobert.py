"""
PhoBERT Fine-Tuning Pipeline
==============================
Fine-tune PhoBERT-base thành classifier 10 class cho phân tích lỗi thiết bị.

Pipeline:
  1. Thu thập & chuẩn hóa dữ liệu (reuse data_preparation.py)
  2. Tokenize bằng PhoBERT tokenizer
  3. Tạo classification head (Linear 768→10)
  4. Fine-tune với AdamW + warmup scheduler
  5. Evaluate trên validation set
  6. Lưu model artifacts

Usage:
  cd /mnt/atin/QuyNB/project/master_project/nlp
  conda run -n mlops python -m backend.training.train_phobert
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from transformers import AutoModel, AutoTokenizer

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.config import (
    PHOBERT_BATCH_SIZE,
    PHOBERT_DROPOUT,
    PHOBERT_EPOCHS,
    PHOBERT_FINETUNED_DIR,
    PHOBERT_FREEZE_EMBEDDINGS,
    PHOBERT_LR,
    PHOBERT_MAX_LENGTH,
    PHOBERT_NUM_CLASSES,
    PHOBERT_WARMUP_RATIO,
    PHOBERT_WEIGHT_DECAY,
    MIN_F1_THRESHOLD,
    MODEL_VERSION,
    RANDOM_STATE,
)
from backend.training.data_preparation import prepare_training_data


# ============================================================
# 1. DATASET CLASS
# ============================================================
class FaultDataset(Dataset):
    """PyTorch Dataset cho fault classification."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ============================================================
# 2. CLASSIFICATION MODEL
# ============================================================
class PhoBERTClassifier(nn.Module):
    """PhoBERT + Linear classification head."""

    def __init__(self, model_path, num_classes=10, dropout=0.3):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_path)
        hidden_size = self.phobert.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # [CLS] token embedding
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

    def freeze_embeddings(self):
        """Freeze embedding layer to save memory & prevent overfitting."""
        for param in self.phobert.embeddings.parameters():
            param.requires_grad = False
        print("  ❄️  Froze PhoBERT embedding layer")

    def get_trainable_params(self):
        """Count trainable vs total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total


# ============================================================
# 3. TRAINING LOOP
# ============================================================
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """Train 1 epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1, all_preds, all_labels


# ============================================================
# 4. MAIN TRAINING PIPELINE
# ============================================================
def main():
    print("=" * 60)
    print("  🚀 PhoBERT Fine-Tuning Pipeline")
    print(f"  Version: {MODEL_VERSION}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    t_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # ============================================================
    # STEP 1: Data preparation
    # ============================================================
    texts, labels_str = prepare_training_data()

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels_str)
    num_classes = len(label_encoder.classes_)
    print(f"\n  Classes ({num_classes}): {list(label_encoder.classes_)}")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    print(f"  Train: {len(X_train)} | Val: {len(X_val)}")

    # ============================================================
    # STEP 2: Load PhoBERT tokenizer & model
    # ============================================================
    print("\n🔧 Step 2: Loading PhoBERT model...")

    # Try local first, then HuggingFace
    local_path = Path(__file__).resolve().parent.parent / "resources" / "phobert-base"
    if local_path.exists():
        model_path = str(local_path)
        print(f"  📂 Loading from local: {model_path}")
    else:
        model_path = "vinai/phobert-base"
        print(f"  🌐 Loading from HuggingFace: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = PhoBERTClassifier(
        model_path=model_path,
        num_classes=num_classes,
        dropout=PHOBERT_DROPOUT,
    ).to(device)

    # Freeze embeddings if configured
    if PHOBERT_FREEZE_EMBEDDINGS:
        model.freeze_embeddings()

    trainable, total = model.get_trainable_params()
    print(f"  Parameters: {trainable:,} trainable / {total:,} total ({trainable/total*100:.1f}%)")

    # ============================================================
    # STEP 3: Create DataLoaders
    # ============================================================
    print("\n🔧 Step 3: Creating datasets...")
    train_dataset = FaultDataset(X_train, y_train, tokenizer, PHOBERT_MAX_LENGTH)
    val_dataset = FaultDataset(X_val, y_val, tokenizer, PHOBERT_MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset, batch_size=PHOBERT_BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=PHOBERT_BATCH_SIZE, shuffle=False,
    )
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ============================================================
    # STEP 4: Optimizer & Scheduler
    # ============================================================
    print("\n🔧 Step 4: Setting up optimizer & scheduler...")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHOBERT_LR,
        weight_decay=PHOBERT_WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * PHOBERT_EPOCHS
    warmup_steps = int(total_steps * PHOBERT_WARMUP_RATIO)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    # After warmup, use cosine decay
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps),
        ],
        milestones=[warmup_steps],
    )

    criterion = nn.CrossEntropyLoss()
    print(f"  Epochs: {PHOBERT_EPOCHS}")
    print(f"  LR: {PHOBERT_LR}")
    print(f"  Warmup steps: {warmup_steps}/{total_steps}")

    # ============================================================
    # STEP 5: Training
    # ============================================================
    print("\n🔥 Step 5: Training...")
    print("-" * 70)
    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'Val F1':>6}")
    print("-" * 70)

    best_val_f1 = 0.0
    best_epoch = 0
    patience = 5
    patience_counter = 0
    best_state = None

    for epoch in range(1, PHOBERT_EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
        )
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device,
        )

        print(f"  {epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.1%} | {val_loss:>8.4f} | {val_acc:>6.1%} | {val_f1:>6.4f}")

        # Track best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\n  ⏹️  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print("-" * 70)
    print(f"  🏆 Best epoch: {best_epoch} | Val F1: {best_val_f1:.4f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # ============================================================
    # STEP 6: Final evaluation
    # ============================================================
    print("\n📊 Step 6: Final evaluation...")
    _, final_acc, final_f1, final_preds, final_labels = evaluate(
        model, val_loader, criterion, device,
    )

    report = classification_report(
        final_labels, final_preds,
        target_names=label_encoder.classes_,
        output_dict=True,
    )
    report_text = classification_report(
        final_labels, final_preds,
        target_names=label_encoder.classes_,
    )
    print(report_text)

    if final_f1 < MIN_F1_THRESHOLD:
        print(f"\n  ⚠️  WARNING: F1={final_f1:.4f} < threshold={MIN_F1_THRESHOLD}")
    else:
        print(f"\n  ✅ F1={final_f1:.4f} ≥ threshold={MIN_F1_THRESHOLD} — PASSED")

    # ============================================================
    # STEP 7: Save model artifacts
    # ============================================================
    print("\n💾 Step 7: Saving model artifacts...")

    PHOBERT_FINETUNED_DIR.mkdir(parents=True, exist_ok=True)

    # Save classifier head weights
    head_state = {
        "classifier_weight": model.classifier.weight.cpu(),
        "classifier_bias": model.classifier.bias.cpu(),
        "dropout_p": PHOBERT_DROPOUT,
        "num_classes": num_classes,
        "label_classes": list(label_encoder.classes_),
    }
    head_path = PHOBERT_FINETUNED_DIR / "classifier_head.pt"
    torch.save(head_state, head_path)
    print(f"  ✅ {head_path}")

    # Save full fine-tuned model
    model_save_path = PHOBERT_FINETUNED_DIR / "model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"  ✅ {model_save_path}")

    # Save label encoder classes
    import joblib
    le_path = PHOBERT_FINETUNED_DIR / "label_encoder.pkl"
    joblib.dump(label_encoder, le_path)
    print(f"  ✅ {le_path}")

    # ============================================================
    # STEP 8: Metadata
    # ============================================================
    print("\n📋 Step 8: Saving metadata...")
    elapsed = time.perf_counter() - t_start

    metadata = {
        "version": MODEL_VERSION,
        "trained_at": datetime.now().isoformat(),
        "training_time_seconds": round(elapsed, 2),
        "device": str(device),
        "base_model": model_path,
        "config": {
            "epochs": PHOBERT_EPOCHS,
            "best_epoch": best_epoch,
            "learning_rate": PHOBERT_LR,
            "batch_size": PHOBERT_BATCH_SIZE,
            "warmup_ratio": PHOBERT_WARMUP_RATIO,
            "weight_decay": PHOBERT_WEIGHT_DECAY,
            "dropout": PHOBERT_DROPOUT,
            "max_length": PHOBERT_MAX_LENGTH,
            "freeze_embeddings": PHOBERT_FREEZE_EMBEDDINGS,
        },
        "data": {
            "total_samples": len(texts),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "num_classes": num_classes,
            "classes": list(label_encoder.classes_),
        },
        "evaluation": {
            "accuracy": round(final_acc, 4),
            "f1_macro": round(final_f1, 4),
            "best_val_f1": round(best_val_f1, 4),
            "per_class": {
                cls: {
                    "precision": round(report[cls]["precision"], 4),
                    "recall": round(report[cls]["recall"], 4),
                    "f1-score": round(report[cls]["f1-score"], 4),
                }
                for cls in label_encoder.classes_
                if cls in report
            },
        },
    }

    metadata_path = PHOBERT_FINETUNED_DIR / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {metadata_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  ✅ PHOBERT FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"  Version:     {MODEL_VERSION}")
    print(f"  Device:      {device}")
    print(f"  Best epoch:  {best_epoch}/{PHOBERT_EPOCHS}")
    print(f"  Accuracy:    {final_acc:.4f}")
    print(f"  F1 Macro:    {final_f1:.4f}")
    print(f"  Total time:  {elapsed:.2f}s")
    print(f"  Output dir:  {PHOBERT_FINETUNED_DIR}")
    print("=" * 60)

    return metadata


if __name__ == "__main__":
    main()
