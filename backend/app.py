"""
FastAPI Backend — Equipment Fault Analysis with PhoBERT NLP.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.database import database as db
from backend.model.schemas import AnalyzeRequest, AnalyzeResponse, HistoryResponse
from backend.core.nlp_engine import analyze as nlp_analyze

app = FastAPI(title="Equipment Fault Analysis API (PhoBERT NLP)")

# Cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- NLP Analysis ----------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Phân tích mô tả thiết bị bằng PhoBERT NLP pipeline.
    Input: equipment (loại thiết bị) + description (mô tả tự nhiên tiếng Việt)
    Output: fault_type, severity, keywords, recommendations, summary, pipeline_steps
    """
    # Gọi NLP Engine (PhoBERT)
    result = nlp_analyze(req.equipment, req.description)

    # Save to database
    record_id = db.save_analysis(
        equipment=req.equipment,
        description=req.description,
        fault_type=result.fault_type,
        severity=result.severity,
        severity_score=result.severity_score,
        confidence=result.confidence,
        keywords=result.keywords,
        recommendations=result.recommendations,
        summary=result.summary,
    )

    return AnalyzeResponse(
        id=record_id,
        fault_type=result.fault_type,
        severity=result.severity,
        severity_score=result.severity_score,
        confidence=result.confidence,
        keywords=result.keywords,
        recommendations=result.recommendations,
        summary=result.summary,
        pipeline_steps=result.pipeline_steps,
    )


# ---------- History APIs ----------
@app.get("/history")
def get_history(limit: int = 100, offset: int = 0):
    """Get all analysis history."""
    records = db.get_all_history(limit=limit, offset=offset)
    total = db.get_history_count()
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": records
    }


@app.get("/history/{record_id}")
def get_history_detail(record_id: int):
    """Get a specific history record."""
    record = db.get_history_by_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return record


@app.delete("/history/{record_id}")
def delete_history_record(record_id: int):
    """Delete a specific history record."""
    success = db.delete_history(record_id)
    if not success:
        raise HTTPException(status_code=404, detail="Record not found")
    return {"message": "Record deleted successfully", "id": record_id}


@app.delete("/history")
def clear_history():
    """Delete all history records."""
    deleted_count = db.clear_all_history()
    return {"message": f"Deleted {deleted_count} records"}
