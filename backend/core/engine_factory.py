"""
Engine Factory — Lazy singleton factory cho NLP engines.
========================================================
Mỗi engine chỉ khởi tạo 1 lần khi được request lần đầu.
"""
from backend.core.base_engine import BaseNLPEngine


_engines: dict[str, BaseNLPEngine] = {}


def get_engine(name: str = "phobert") -> BaseNLPEngine:
    """
    Lấy engine instance theo tên. Lazy init + singleton.

    Args:
        name: "phobert" | "tfidf"

    Returns:
        BaseNLPEngine instance

    Raises:
        ValueError: nếu engine name không hợp lệ
    """
    if name not in _engines:
        if name == "phobert":
            from backend.core.phobert_engine import PhoBERTEngine
            _engines[name] = PhoBERTEngine()
        elif name == "tfidf":
            from backend.core.tfidf_engine import TFIDFEngine
            _engines[name] = TFIDFEngine()
        else:
            raise ValueError(f"Unknown engine: '{name}'. Available: phobert, tfidf")
    return _engines[name]


def list_engines() -> list[str]:
    """Trả về danh sách tên engine có sẵn."""
    return ["phobert", "tfidf"]
