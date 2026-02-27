"""
FastAPI Backend — Equipment Fault Analysis with Multi-Engine NLP.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.database import database as db
from backend.model.schemas import AnalyzeRequest, AnalyzeResponse, CompareResponse
from backend.core.engine_factory import get_engine, list_engines

app = FastAPI(title="Equipment Fault Analysis API (Multi-Engine NLP)")

# Cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _result_to_response(result, record_id=None) -> AnalyzeResponse:
    """Chuyển AnalysisResult → AnalyzeResponse."""
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
        engine_name=result.engine_name,
        engine_latency_ms=result.engine_latency_ms,
    )


# ---------- NLP Analysis ----------
@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """
    Phân tích mô tả thiết bị bằng NLP pipeline.
    Hỗ trợ: engine="phobert" | "tfidf" | "compare"
    """
    if req.engine == "compare":
        # So sánh cả 2 engine
        result_phobert = get_engine("phobert").analyze(req.equipment, req.description)
        result_tfidf = get_engine("tfidf").analyze(req.equipment, req.description)

        # Save PhoBERT result to DB (primary)
        record_id = db.save_analysis(
            equipment=req.equipment,
            description=req.description,
            fault_type=result_phobert.fault_type,
            severity=result_phobert.severity,
            severity_score=result_phobert.severity_score,
            confidence=result_phobert.confidence,
            keywords=result_phobert.keywords,
            recommendations=result_phobert.recommendations,
            summary=result_phobert.summary,
            engine_name="compare",
        )

        return CompareResponse(
            phobert=_result_to_response(result_phobert, record_id),
            tfidf=_result_to_response(result_tfidf),
        )
    else:
        # Single engine
        engine = get_engine(req.engine)
        result = engine.analyze(req.equipment, req.description)

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
            engine_name=result.engine_name,
        )

        return _result_to_response(result, record_id)


# ---------- Engine Info ----------
@app.get("/engines")
def get_available_engines():
    """Trả về danh sách engine có sẵn."""
    return {"engines": list_engines()}


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


# ---------- Evaluation API ----------
@app.get("/evaluation/summary")
def evaluation_summary():
    """
    Get evaluation summary (accuracy, macro_f1, avg_latency_ms).
    Reads from cached report file if available.
    """
    import json
    from pathlib import Path

    report_path = Path(__file__).resolve().parent / "evaluation" / "results" / "evaluation_report.json"

    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation report not found. Run: python -m backend.evaluation.run_evaluation"
        )

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    return {
        "accuracy": report.get("accuracy", 0),
        "macro_f1": report.get("f1_macro", 0),
        "avg_latency_ms": report.get("latency", {}).get("mean_ms", 0),
        "precision_macro": report.get("precision_macro", 0),
        "recall_macro": report.get("recall_macro", 0),
        "per_class": report.get("per_class", {}),
    }
