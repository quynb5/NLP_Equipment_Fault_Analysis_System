"""
Phân tích chi tiết: PhoBERT đúng nhưng TF-IDF sai (và ngược lại).
Dùng evaluation_dataset.json (149 mẫu).
"""
import json
import sys
sys.path.insert(0, ".")
from backend.core.phobert_engine import PhoBERTEngine
from backend.core.tfidf_engine import TFIDFEngine

with open("backend/evaluation/evaluation_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

phobert = PhoBERTEngine()
tfidf = TFIDFEngine()

phobert_wins = []
tfidf_wins = []

for i, item in enumerate(data):
    desc = item["description"]
    expected = item["true_label"]
    r_pho = phobert.analyze("TB", desc)
    r_tfi = tfidf.analyze("TB", desc)

    pho_ok = r_pho.fault_type == expected
    tfi_ok = r_tfi.fault_type == expected

    if pho_ok and not tfi_ok:
        phobert_wins.append((i, desc, expected, r_pho.fault_type, r_pho.confidence, r_tfi.fault_type, r_tfi.confidence))
    elif tfi_ok and not pho_ok:
        tfidf_wins.append((i, desc, expected, r_pho.fault_type, r_pho.confidence, r_tfi.fault_type, r_tfi.confidence))

print("=" * 90)
print(f"  PhoBERT đúng, TF-IDF sai: {len(phobert_wins)} mẫu")
print(f"  TF-IDF đúng, PhoBERT sai: {len(tfidf_wins)} mẫu")
print("=" * 90)

print("\n" + "=" * 90)
print("  🔥 PhoBERT WINS (PhoBERT đúng, TF-IDF sai)")
print("=" * 90)
for idx, desc, exp, pho_pred, pho_conf, tfi_pred, tfi_conf in phobert_wins:
    print(f"\n  [{idx}] \"{desc[:85]}...\"")
    print(f"      Expected:  {exp}")
    print(f"      PhoBERT:   {pho_pred} (conf={pho_conf}) ✅")
    print(f"      TF-IDF:    {tfi_pred} (conf={tfi_conf}) ❌")

print("\n" + "=" * 90)
print("  📊 TF-IDF WINS (TF-IDF đúng, PhoBERT sai)")
print("=" * 90)
for idx, desc, exp, pho_pred, pho_conf, tfi_pred, tfi_conf in tfidf_wins:
    print(f"\n  [{idx}] \"{desc[:85]}...\"")
    print(f"      Expected:  {exp}")
    print(f"      PhoBERT:   {pho_pred} (conf={pho_conf}) ❌")
    print(f"      TF-IDF:    {tfi_pred} (conf={tfi_conf}) ✅")
