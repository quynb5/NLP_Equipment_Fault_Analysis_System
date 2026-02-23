"""
Pydantic models for API requests and responses.
Updated for NLP Engine (PhoBERT) — free-text input.
"""
from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    """Request model — nhận mô tả tự nhiên thay vì dropdown."""
    equipment: str           # Loại thiết bị
    description: str         # Mô tả tự nhiên tiếng Việt


class AnalyzeResponse(BaseModel):
    """Response model — kết quả phân tích NLP."""
    id: int | None = None
    fault_type: str          # Loại lỗi (PhoBERT classification)
    severity: str            # Mức độ: THẤP / CẢNH BÁO / NGHIÊM TRỌNG
    severity_score: float    # Điểm severity (0-1)
    confidence: float        # Confidence score từ PhoBERT
    keywords: list[str]      # Từ khóa phát hiện
    recommendations: list[str]
    summary: str
    pipeline_steps: list[dict] = []  # Pipeline NLP steps (optional)


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
