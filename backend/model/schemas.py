"""
Pydantic models for API requests and responses.
Multi-engine NLP architecture.
"""
from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    """Request model — nhận mô tả tự nhiên + chọn engine."""
    equipment: str           # Loại thiết bị
    description: str         # Mô tả tự nhiên tiếng Việt
    engine: str = "phobert"  # "phobert" | "tfidf" | "compare"


class AnalyzeResponse(BaseModel):
    """Response model — kết quả phân tích NLP."""
    id: int | None = None
    fault_type: str          # Loại lỗi (classification)
    severity: str            # Mức độ: THẤP / CẢNH BÁO / NGHIÊM TRỌNG
    severity_score: float    # Điểm severity (0-1)
    confidence: float        # Confidence score
    keywords: list[str]      # Từ khóa phát hiện
    recommendations: list[str]
    summary: str
    pipeline_steps: list[dict] = []
    engine_name: str = ""         # Engine đã dùng
    engine_latency_ms: float = 0.0  # Latency (ms)


class CompareResponse(BaseModel):
    """Response cho compare mode — kết quả từ 2 engine."""
    phobert: AnalyzeResponse
    tfidf: AnalyzeResponse


class HistoryResponse(BaseModel):
    """Response model for history record."""
    id: int
    created_at: str
    equipment: str
    description: str         # Mô tả gốc
    fault_type: str
    severity: str
    severity_score: float
    confidence: float
    keywords: list[str]
    recommendations: list[str]
    summary: str
    engine_name: str = "phobert"  # Engine đã dùng
