"""Phase 4: End-to-end API verification."""
import requests
import json

BASE = "http://localhost:10805"

print("=" * 60)
print("PHASE 4: END-TO-END VERIFICATION")
print("=" * 60)

# Test 1: Home page
print("\n=== TEST 1: Home page ===")
r = requests.get(f"{BASE}/")
print(f"  GET / → HTTP {r.status_code}")
assert r.status_code == 200, "FAIL"
assert "PhoBERT" in r.text, "FAIL - PhoBERT not in HTML"
print("✅ PASS")

# Test 2: Background image
print("\n=== TEST 2: Background image ===")
r = requests.get(f"{BASE}/resources/images/bg.png")
print(f"  GET /resources/images/bg.png → HTTP {r.status_code}")
assert r.status_code == 200, "FAIL"
print("✅ PASS")

# Test 3: Analyze - severe fault
print("\n=== TEST 3: Analyze - severe fault ===")
r = requests.post(f"{BASE}/analyze", json={
    "equipment": "Motor",
    "description": "Motor nóng bất thường, rung mạnh, nghe tiếng kim loại va chạm"
})
d = r.json()
print(f"  fault_type: {d['fault_type']}")
print(f"  severity: {d['severity']} (score: {d['severity_score']})")
print(f"  confidence: {d['confidence']}")
print(f"  keywords: {d['keywords']}")
print(f"  pipeline_steps: {len(d.get('pipeline_steps', []))} steps")
print(f"  id: {d['id']}")
assert d['severity'] == 'NGHIÊM TRỌNG', f"FAIL: expected NGHIÊM TRỌNG, got {d['severity']}"
assert len(d['keywords']) > 0, "FAIL: no keywords"
assert d['id'] is not None, "FAIL: no DB id"
print("✅ PASS")

# Test 4: Analyze - normal operation
print("\n=== TEST 4: Analyze - normal ===")
r = requests.post(f"{BASE}/analyze", json={
    "equipment": "Bơm",
    "description": "Máy hoạt động bình thường, êm ái, không có tiếng ồn lạ"
})
d = r.json()
print(f"  fault_type: {d['fault_type']}")
print(f"  severity: {d['severity']}")
assert d['fault_type'] == 'Hoạt động ổn định', f"FAIL: got {d['fault_type']}"
assert d['severity'] == 'THẤP', f"FAIL: got {d['severity']}"
print("✅ PASS")

# Test 5: Analyze - fire
print("\n=== TEST 5: Analyze - fire ===")
r = requests.post(f"{BASE}/analyze", json={
    "equipment": "Motor",
    "description": "Mùi khét, bốc khói, dòng điện tăng đột ngột"
})
d = r.json()
print(f"  fault_type: {d['fault_type']}")
print(f"  severity: {d['severity']}")
assert 'Cháy' in d['fault_type'], f"FAIL: got {d['fault_type']}"
assert d['severity'] == 'NGHIÊM TRỌNG', f"FAIL: got {d['severity']}"
print("✅ PASS")

# Test 6: Analyze - leak
print("\n=== TEST 6: Analyze - leak ===")
r = requests.post(f"{BASE}/analyze", json={
    "equipment": "Máy nén",
    "description": "Rò rỉ dầu, gioăng mòn nhiều"
})
d = r.json()
print(f"  fault_type: {d['fault_type']}")
print(f"  severity: {d['severity']}")
print(f"  keywords: {d['keywords']}")
assert 'Rò rỉ' in d['fault_type'], f"FAIL: got {d['fault_type']}"
print("✅ PASS")

# Test 7: History list
print("\n=== TEST 7: History list ===")
r = requests.get(f"{BASE}/history")
d = r.json()
print(f"  total: {d['total']}")
print(f"  records: {len(d['data'])}")
assert d['total'] >= 4, f"FAIL: expected >= 4 records"
print("✅ PASS")

# Test 8: History detail
print("\n=== TEST 8: History detail ===")
first_id = d['data'][0]['id']
r = requests.get(f"{BASE}/history/{first_id}")
d = r.json()
print(f"  equipment: {d['equipment']}")
print(f"  description: {d['description'][:60]}...")
print(f"  fault_type: {d['fault_type']}")
print(f"  keywords: {d['keywords']}")
assert 'description' in d, "FAIL: no description"
assert 'fault_type' in d, "FAIL: no fault_type"
assert 'keywords' in d, "FAIL: no keywords"
print("✅ PASS")

# Test 9: Delete record
print("\n=== TEST 9: Delete record ===")
r = requests.delete(f"{BASE}/history/{first_id}")
d = r.json()
print(f"  message: {d['message']}")
r2 = requests.get(f"{BASE}/history/{first_id}")
assert r2.status_code == 404, "FAIL: record not deleted"
print("✅ PASS")

# Test 10: Clear all
print("\n=== TEST 10: Clear all history ===")
r = requests.delete(f"{BASE}/history")
d = r.json()
print(f"  message: {d['message']}")
r2 = requests.get(f"{BASE}/history")
assert r2.json()['total'] == 0, "FAIL: not cleared"
print("✅ PASS")

print("\n" + "=" * 60)
print("✅ ALL 10 TESTS PASSED — PHASE 4 COMPLETE")
print("=" * 60)
