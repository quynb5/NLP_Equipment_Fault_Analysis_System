"""
TASK 02 — Prediction Wrapper
Gọi nlp_engine.analyze() và trả về predicted_label.
"""
import sys
from pathlib import Path

# Thêm backend/core vào path để import nlp_engine
_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR / "core"))

from backend.core.nlp_engine import analyze


def predict_label(text: str) -> str:
    """
    Gọi NLP Engine và trả về predicted fault_type.

    Args:
        text: Mô tả tình trạng thiết bị bằng tiếng Việt

    Returns:
        Predicted fault type string
    """
    result = analyze(equipment="Thiết bị", description=text)
    return result.fault_type
