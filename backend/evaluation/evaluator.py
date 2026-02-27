"""
TASK 02 — Prediction Wrapper
Gọi engine.analyze() và trả về predicted_label.
Hỗ trợ multi-engine qua engine_factory.
"""
import sys
from pathlib import Path

# Thêm project root vào path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from backend.core.engine_factory import get_engine


def predict_label(text: str, engine_name: str = "phobert") -> str:
    """
    Gọi NLP Engine và trả về predicted fault_type.

    Args:
        text: Mô tả tình trạng thiết bị bằng tiếng Việt
        engine_name: "phobert" hoặc "tfidf"

    Returns:
        Predicted fault type string
    """
    engine = get_engine(engine_name)
    result = engine.analyze(equipment="Thiết bị", description=text)
    return result.fault_type
