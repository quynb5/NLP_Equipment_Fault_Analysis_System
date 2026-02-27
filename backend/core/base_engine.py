"""
Base NLP Engine — Abstract interface cho multi-engine architecture.
===================================================================
Mọi engine (PhoBERT, TF-IDF, ...) đều kế thừa BaseNLPEngine
và trả về AnalysisResult chuẩn.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ============================================================
# ANALYSIS RESULT (dùng chung cho mọi engine)
# ============================================================

@dataclass
class AnalysisResult:
    """Kết quả phân tích NLP — chuẩn hóa cho mọi engine."""
    fault_type: str
    severity: str
    severity_score: float
    confidence: float
    keywords: list = field(default_factory=list)
    symptoms: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    summary: str = ""
    pipeline_steps: list = field(default_factory=list)
    engine_name: str = ""            # Tên engine đã dùng (vd: "phobert", "tfidf")
    engine_latency_ms: float = 0.0   # Thời gian inference (ms)


# ============================================================
# BASE ENGINE INTERFACE
# ============================================================

class BaseNLPEngine(ABC):
    """
    Interface chung cho mọi NLP engine.

    Mỗi engine cần implement:
      - name (property): tên định danh engine
      - analyze(): pipeline phân tích chính
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tên engine, vd: 'phobert', 'tfidf'."""
        ...

    @abstractmethod
    def analyze(self, equipment: str, description: str) -> AnalysisResult:
        """
        Phân tích mô tả thiết bị và trả về kết quả.

        Args:
            equipment: Loại thiết bị (Motor, Bơm, ...)
            description: Mô tả tự nhiên tiếng Việt

        Returns:
            AnalysisResult chứa fault_type, severity, keywords, ...
        """
        ...
