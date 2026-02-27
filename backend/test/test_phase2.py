"""Quick verification of Phase 2 backend refactor."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.model.schemas import AnalyzeRequest, AnalyzeResponse, HistoryResponse, CompareResponse
print("✅ Schemas imported OK")
print("  AnalyzeRequest:", list(AnalyzeRequest.model_fields.keys()))
print("  AnalyzeResponse:", list(AnalyzeResponse.model_fields.keys()))
print("  CompareResponse:", list(CompareResponse.model_fields.keys()))

from backend.database import database as db
print("✅ Database imported OK")

from backend.core.engine_factory import get_engine
engine = get_engine("phobert")
print(f"✅ Engine Factory OK — loaded engine: {engine.name}")

# Test NLP → DB pipeline
result = engine.analyze("Motor khu A", "Motor nóng bất thường, rung mạnh")
print(f"✅ NLP: fault={result.fault_type} | severity={result.severity} | engine={result.engine_name} | latency={result.engine_latency_ms}ms")

rid = db.save_analysis(
    equipment="Motor khu A",
    description="Motor nóng bất thường, rung mạnh",
    fault_type=result.fault_type,
    severity=result.severity,
    severity_score=result.severity_score,
    confidence=result.confidence,
    keywords=result.keywords,
    recommendations=result.recommendations,
    summary=result.summary,
    engine_name=result.engine_name,
)
print(f"✅ DB save OK, id={rid}")

rec = db.get_history_by_id(rid)
print(f"✅ DB read OK: {rec['fault_type']} | kw={rec['keywords']}")

print("\n✅ ALL PHASE 2 CHECKS PASSED")
