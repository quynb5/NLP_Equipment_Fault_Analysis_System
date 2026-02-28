"""
Benchmark: PhoBERT vs TF-IDF — Các trường hợp TF-IDF bị kém
============================================================
Mục đích: Chứng minh ưu điểm semantic understanding của PhoBERT
so với keyword-matching của TF-IDF cho báo cáo học thuật.
"""
import json
import sys
sys.path.insert(0, ".")

from backend.core.phobert_engine import PhoBERTEngine
from backend.core.tfidf_engine import TFIDFEngine

# ============================================================
# TEST CASES — Thiết kế để expose điểm yếu của TF-IDF
# ============================================================

TEST_CASES = [
    # === NHÓM 1: PARAPHRASING (cùng nghĩa, khác từ) ===
    # TF-IDF dựa vào keyword → fail khi dùng từ đồng nghĩa không có trong training
    {
        "category": "Paraphrasing",
        "description": "Thiết bị phát ra âm lượng lớn bất thường khi vận hành, có tiếng lạch cạch từ bên trong",
        "expected": "Âm thanh bất thường",
        "reason": "TF-IDF không có feature 'âm lượng lớn' hay 'lạch cạch' → miss. PhoBERT hiểu ngữ nghĩa 'âm lượng lớn' ≈ 'tiếng ồn'"
    },
    {
        "category": "Paraphrasing", 
        "description": "Máy nén tiêu thụ điện năng nhiều hơn 30% so với thông số kỹ thuật, output giảm rõ rệt",
        "expected": "Giảm hiệu suất",
        "reason": "TF-IDF không map 'tiêu thụ điện năng nhiều hơn' → giảm hiệu suất. PhoBERT hiểu context tổng thể"
    },
    {
        "category": "Paraphrasing",
        "description": "Trục quay bị cong vênh, gây mài mòn và nghiến kim loại khi chạy tải nặng",
        "expected": "Hư hỏng cơ khí",
        "reason": "TF-IDF yếu với các biến thể mô tả cơ khí. PhoBERT hiểu 'cong vênh', 'mài mòn', 'nghiến' → cơ khí"
    },

    # === NHÓM 2: MÔ TẢ DÀI VÀ PHỨC TẠP (multi-symptom) ===
    # TF-IDF bị phân tán bởi nhiều keyword → chọn sai class
    {
        "category": "Complex Multi-symptom",
        "description": "Sau khi chạy liên tục 8 giờ, bề mặt vỏ máy nóng đến mức không chạm tay vào được, quạt tản nhiệt vẫn quay nhưng không đủ gió, nhiệt kế hồng ngoại đo được 95°C",
        "expected": "Quá nhiệt",
        "reason": "Câu dài chứa nhiều chi tiết kỹ thuật. TF-IDF bị phân tán bởi nhiều features không related. PhoBERT hiểu toàn bộ context"
    },
    {
        "category": "Complex Multi-symptom",
        "description": "Chất lỏng thủy lực rỉ ra ở mối nối, tạo thành vũng dầu dưới sàn, áp suất hệ thống giảm từ 150 bar xuống còn 80 bar chỉ sau 2 giờ vận hành",
        "expected": "Rò rỉ hệ thống",
        "reason": "Mô tả kỹ thuật chi tiết với số liệu. TF-IDF miss nếu 'chất lỏng thủy lực rỉ' không match training data. PhoBERT hiểu semantic"
    },

    # === NHÓM 3: NGỮ CẢNH GIÁN TIẾP (indirect description) ===
    # Mô tả triệu chứng gián tiếp, không dùng keyword trực tiếp
    {
        "category": "Indirect Description",
        "description": "Đồng hồ ampe kế chỉ gấp đôi bình thường, relay nhiệt nhảy liên tục, phải reset mỗi 15 phút",
        "expected": "Quá tải cơ khí",
        "reason": "Không có từ 'quá tải' trực tiếp. TF-IDF miss. PhoBERT hiểu 'ampe gấp đôi + relay nhiệt nhảy' = quá tải"
    },
    {
        "category": "Indirect Description",
        "description": "Khe hở giữa rotor và stator không đều, một bên chật một bên rộng, gây tiếng cọ kim loại khi quay",
        "expected": "Hư hỏng cơ khí",
        "reason": "Mô tả chuyên ngành sâu. TF-IDF không có feature 'rotor stator khe hở'. PhoBERT encode ngữ cảnh tổng thể"
    },

    # === NHÓM 4: VIẾT TẮT / THUẬT NGỮ CHUYÊN NGÀNH ===
    {
        "category": "Technical Jargon",
        "description": "Megger test đo IR cuộn dây cho giá trị 0.5MΩ, dưới ngưỡng an toàn 1MΩ, nguy cơ chạm mass",
        "expected": "Cháy cuộn dây / cháy motor",
        "reason": "Thuật ngữ chuyên ngành (Megger, IR, MΩ, chạm mass). TF-IDF không có features này. PhoBERT hiểu context 'cuộn dây + nguy cơ'"
    },
    {
        "category": "Technical Jargon",
        "description": "VFD báo lỗi overcurrent, kiểm tra thấy một pha bị mất, contactor 3 pha chỉ đóng được 2 pha",
        "expected": "Sự cố điện",
        "reason": "Viết tắt VFD, overcurrent, pha. TF-IDF miss viết tắt tiếng Anh. PhoBERT hiểu mixed-language context"
    },

    # === NHÓM 5: CÂU NGẮN VỚI ÍT THÔNG TIN ===
    {
        "category": "Short & Ambiguous",
        "description": "Máy bị nặng tay khi vận hành",
        "expected": "Quá tải cơ khí",
        "reason": "Câu ngắn, ít keyword. TF-IDF không đủ features. PhoBERT hiểu 'nặng tay' = tải nặng/quá tải"
    },
    {
        "category": "Short & Ambiguous",
        "description": "Công suất ra không đạt như trước",
        "expected": "Giảm hiệu suất",
        "reason": "Câu rất ngắn. TF-IDF miss vì 'công suất ra không đạt' có thể không match features. PhoBERT hiểu semantic"
    },
]

def main():
    print("=" * 80)
    print("  BENCHMARK: PhoBERT (Fine-tuned) vs TF-IDF")
    print("  Các trường hợp TF-IDF bị kém hẳn")
    print("=" * 80)

    phobert = PhoBERTEngine()
    tfidf = TFIDFEngine()

    phobert_correct = 0
    tfidf_correct = 0
    results = []

    for i, tc in enumerate(TEST_CASES):
        r_pho = phobert.analyze("Thiết bị", tc["description"])
        r_tfi = tfidf.analyze("Thiết bị", tc["description"])

        pho_ok = r_pho.fault_type == tc["expected"]
        tfi_ok = r_tfi.fault_type == tc["expected"]
        phobert_correct += pho_ok
        tfidf_correct += tfi_ok

        status = ""
        if pho_ok and not tfi_ok:
            status = "✅ PhoBERT WIN"
        elif not pho_ok and tfi_ok:
            status = "❌ TF-IDF WIN"
        elif pho_ok and tfi_ok:
            status = "🔵 BOTH CORRECT"
        else:
            status = "⚫ BOTH WRONG"

        results.append({
            "index": i + 1,
            "category": tc["category"],
            "status": status,
            "expected": tc["expected"],
            "phobert_pred": r_pho.fault_type,
            "phobert_conf": r_pho.confidence,
            "tfidf_pred": r_tfi.fault_type,
            "tfidf_conf": r_tfi.confidence,
            "reason": tc["reason"],
        })

    print()
    print("=" * 80)
    print("  KẾT QUẢ CHI TIẾT")
    print("=" * 80)

    for r in results:
        print(f"\n{'─' * 70}")
        print(f"  [{r['index']}] {r['category']}  |  {r['status']}")
        print(f"  Input: \"{TEST_CASES[r['index']-1]['description'][:70]}...\"")
        print(f"  Expected: {r['expected']}")
        print(f"  PhoBERT:  {r['phobert_pred']} (conf={r['phobert_conf']})  {'✅' if r['phobert_pred'] == r['expected'] else '❌'}")
        print(f"  TF-IDF:   {r['tfidf_pred']} (conf={r['tfidf_conf']})  {'✅' if r['tfidf_pred'] == r['expected'] else '❌'}")
        print(f"  📝 {r['reason']}")

    print(f"\n{'=' * 80}")
    print(f"  TỔNG KẾT")
    print(f"{'=' * 80}")
    print(f"  PhoBERT: {phobert_correct}/{len(TEST_CASES)} ({100*phobert_correct/len(TEST_CASES):.0f}%)")
    print(f"  TF-IDF:  {tfidf_correct}/{len(TEST_CASES)} ({100*tfidf_correct/len(TEST_CASES):.0f}%)")
    
    pho_wins = sum(1 for r in results if "PhoBERT WIN" in r["status"])
    tfi_wins = sum(1 for r in results if "TF-IDF WIN" in r["status"])
    both_ok = sum(1 for r in results if "BOTH CORRECT" in r["status"])
    both_wrong = sum(1 for r in results if "BOTH WRONG" in r["status"])
    
    print(f"\n  PhoBERT wins: {pho_wins}")
    print(f"  TF-IDF wins:  {tfi_wins}")
    print(f"  Both correct: {both_ok}")
    print(f"  Both wrong:   {both_wrong}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
