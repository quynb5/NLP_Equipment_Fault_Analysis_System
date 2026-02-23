"""Quick verification of Phase 2 backend refactor."""
from model.schemas import AnalyzeRequest, AnalyzeResponse, HistoryResponse
print("✅ Schemas imported OK")
print("  AnalyzeRequest:", list(AnalyzeRequest.model_fields.keys()))
print("  AnalyzeResponse:", list(AnalyzeResponse.model_fields.keys()))

from database import database as db
print("✅ Database imported OK")

from nlp_engine import analyze as nlp_analyze
print("✅ NLP Engine imported OK")

# Test NLP → DB pipeline
result = nlp_analyze("Motor khu A", "Motor nóng bất thường, rung mạnh")
print(f"✅ NLP: fault={result.fault_type} | severity={result.severity}")

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
)
print(f"✅ DB save OK, id={rid}")

rec = db.get_history_by_id(rid)
print(f"✅ DB read OK: {rec['fault_type']} | kw={rec['keywords']}")

print("\n✅ ALL PHASE 2 CHECKS PASSED")
