"""
Data Preparation — Thu thập & augment dữ liệu training.
=========================================================
3 nguồn dữ liệu:
  Source 1: FAULT_REFERENCES samples từ phobert_engine.py
  Source 2: test_dataset.json (labeled)
  Source 3: Data augmentation (đảo từ, thay từ đồng nghĩa)
"""
import json
import random
import re
from pathlib import Path

from backend.training.config import (
    TEST_DATASET_PATH,
    AUGMENT_ENABLED,
    AUGMENT_MULTIPLIER,
    RANDOM_STATE,
)

random.seed(RANDOM_STATE)


# ========================
# LABEL NORMALIZATION
# ========================
# FAULT_REFERENCES dùng "Bình thường",
# test_dataset dùng "Hoạt động ổn định".
# Chuẩn hóa về label thống nhất (dùng test_dataset convention).
LABEL_MAP = {
    "Bình thường": "Hoạt động ổn định",
}


def normalize_label(label: str) -> str:
    """Chuẩn hóa label về tên thống nhất."""
    return LABEL_MAP.get(label, label)


# ========================
# SOURCE 1: FAULT_REFERENCES
# ========================
def load_fault_references() -> list[dict]:
    """
    Extract samples từ FAULT_REFERENCES trong phobert_engine.py.
    Return: [{"description": ..., "label": ...}, ...]
    """
    from backend.core.phobert_engine import FAULT_REFERENCES

    data = []
    for fault_type, fault_data in FAULT_REFERENCES.items():
        label = normalize_label(fault_type)
        for sample in fault_data["samples"]:
            data.append({"description": sample, "label": label})

    print(f"  📚 Source 1 (FAULT_REFERENCES): {len(data)} samples")
    return data


# ========================
# SOURCE 2: TEST DATASET
# ========================
def load_test_dataset() -> list[dict]:
    """
    Load test_dataset.json.
    Return: [{"description": ..., "label": ...}, ...]
    """
    with open(TEST_DATASET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = [
        {"description": item["description"], "label": normalize_label(item["true_label"])}
        for item in raw
    ]
    print(f"  📚 Source 2 (test_dataset.json): {len(data)} samples")
    return data


# ========================
# SOURCE 3: DATA AUGMENTATION
# ========================

# Từ đồng nghĩa cho thiết bị công nghiệp
SYNONYM_MAP = {
    "motor": ["động cơ", "máy", "mô tơ"],
    "động cơ": ["motor", "máy", "mô tơ"],
    "thiết bị": ["máy móc", "máy", "hệ thống"],
    "máy": ["thiết bị", "motor", "hệ thống"],
    "nóng": ["nhiệt cao", "nóng bất thường", "phát nhiệt"],
    "rung": ["rung lắc", "rung động", "dao động"],
    "rung mạnh": ["rung lắc dữ dội", "rung động mạnh"],
    "tiếng kêu": ["âm thanh", "tiếng ồn", "tiếng động"],
    "hỏng": ["hư hỏng", "bị hỏng", "trục trặc"],
    "rò rỉ": ["chảy", "rỉ", "xì"],
    "cháy": ["bốc cháy", "cháy khét"],
    "khói": ["bốc khói", "khói bốc"],
    "mùi khét": ["mùi cháy", "mùi cháy khét"],
    "bất thường": ["lạ", "khác thường", "không bình thường"],
    "nghiêm trọng": ["nặng", "trầm trọng", "nguy hiểm"],
    "dầu": ["dầu nhớt", "dầu bôi trơn", "dầu thủy lực"],
    "gioăng": ["phớt", "seal", "roăng"],
    "bu lông": ["ốc vít", "bulông"],
    "dây đai": ["đai truyền động", "belt"],
    "bạc đạn": ["vòng bi", "ổ bi", "bearing"],
    "vòng bi": ["bạc đạn", "ổ bi"],
    "chập điện": ["chập mạch", "ngắn mạch"],
    "quá tải": ["vượt tải", "overload"],
}


def _augment_swap_words(text: str) -> str:
    """Đảo vị trí 2 từ ngẫu nhiên trong câu."""
    words = text.split()
    if len(words) < 3:
        return text
    i, j = random.sample(range(len(words)), 2)
    words[i], words[j] = words[j], words[i]
    return " ".join(words)


def _augment_synonym(text: str) -> str:
    """Thay 1 từ bằng từ đồng nghĩa."""
    words = text.split()
    # Tìm từ/cụm từ có thể thay
    candidates = []
    for key in SYNONYM_MAP:
        if key in text:
            candidates.append(key)

    if not candidates:
        return _augment_swap_words(text)  # Fallback: đảo từ

    target = random.choice(candidates)
    replacement = random.choice(SYNONYM_MAP[target])
    return text.replace(target, replacement, 1)


def augment_data(data: list[dict], multiplier: int = 2) -> list[dict]:
    """
    Tạo dữ liệu augmented.
    Mỗi sample gốc → thêm `multiplier` samples (đảo từ + thay từ đồng nghĩa).
    """
    if not AUGMENT_ENABLED:
        return []

    augmented = []
    for item in data:
        for _ in range(multiplier):
            method = random.choice([_augment_swap_words, _augment_synonym])
            new_text = method(item["description"])
            if new_text != item["description"]:
                augmented.append({
                    "description": new_text,
                    "label": item["label"],
                })

    print(f"  📚 Source 3 (Augmentation ×{multiplier}): {len(augmented)} samples")
    return augmented


# ========================
# MAIN: COLLECT ALL DATA
# ========================
def prepare_training_data() -> tuple[list[str], list[str]]:
    """
    Thu thập & gộp toàn bộ dữ liệu huấn luyện.

    Returns:
        (texts, labels) — parallel lists
    """
    print("\n📦 Preparing training data...")

    # Source 1
    source1 = load_fault_references()

    # Source 2
    source2 = load_test_dataset()

    # Source 3: augment từ cả source 1 + source 2
    source3 = augment_data(source1 + source2, multiplier=AUGMENT_MULTIPLIER)

    # Gộp tất cả
    all_data = source1 + source2 + source3

    # Shuffle
    random.shuffle(all_data)

    texts = [d["description"] for d in all_data]
    labels = [d["label"] for d in all_data]

    # Stats
    from collections import Counter
    label_counts = Counter(labels)
    print(f"\n  📊 Tổng: {len(texts)} samples, {len(label_counts)} classes")
    for label, count in sorted(label_counts.items()):
        print(f"     {label}: {count}")

    return texts, labels
