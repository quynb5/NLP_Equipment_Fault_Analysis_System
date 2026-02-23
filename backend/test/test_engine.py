"""Test PhoBERT NLP Engine."""
from nlp_engine import analyze

print("=" * 60)
print("TEST PhoBERT NLP Engine")
print("=" * 60)

print("\n=== TEST 1: Severe multi-symptom ===")
r1 = analyze("Motor khu A", "Motor khu A nóng bất thường, rung mạnh, nghe tiếng kim loại va chạm")
print(f"Fault: {r1.fault_type}")
print(f"Severity: {r1.severity} (score: {r1.severity_score})")
print(f"Confidence: {r1.confidence}")
print(f"Keywords: {r1.keywords}")

print("\n=== TEST 2: Normal ===")
r2 = analyze("Bơm nước", "Bơm nước hoạt động bình thường, không có tiếng ồn lạ")
print(f"Fault: {r2.fault_type}")
print(f"Severity: {r2.severity} (score: {r2.severity_score})")
print(f"Confidence: {r2.confidence}")
print(f"Keywords: {r2.keywords}")

print("\n=== TEST 3: Fire ===")
r3 = analyze("Máy nén", "Máy nén có mùi khét, dòng điện tăng đột ngột, bốc khói")
print(f"Fault: {r3.fault_type}")
print(f"Severity: {r3.severity} (score: {r3.severity_score})")
print(f"Confidence: {r3.confidence}")
print(f"Keywords: {r3.keywords}")

print("\n=== TEST 4: All negated ===")
r4 = analyze("Quạt", "Quạt không bị rung, không khét, hoạt động ổn định")
print(f"Fault: {r4.fault_type}")
print(f"Severity: {r4.severity} (score: {r4.severity_score})")
print(f"Keywords: {r4.keywords}")

print("\n=== TEST 5: Rò rỉ ===")
r5 = analyze("Bơm thủy lực", "Bơm thủy lực bị rò rỉ dầu, gioăng mòn nhiều")
print(f"Fault: {r5.fault_type}")
print(f"Severity: {r5.severity} (score: {r5.severity_score})")
print(f"Confidence: {r5.confidence}")
print(f"Keywords: {r5.keywords}")

print("\n=== TEST 6: Mixed negation ===")
r6 = analyze("Motor", "Motor không nóng nhưng rung mạnh và có tiếng kêu lạ")
print(f"Fault: {r6.fault_type}")
print(f"Severity: {r6.severity} (score: {r6.severity_score})")
print(f"Confidence: {r6.confidence}")
print(f"Keywords: {r6.keywords}")

print("\n=== TEST 7: Pipeline steps (Test 1) ===")
for step in r1.pipeline_steps:
    out = step["output"]
    if isinstance(out, list) and len(out) > 3:
        out = f"[{len(out)} items]"
    print(f"  Step {step['step']}: {step['name']} -> {out}")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED SUCCESSFULLY")
print("=" * 60)
