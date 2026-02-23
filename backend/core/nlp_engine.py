"""
NLP Engine - Vietnamese Industrial Equipment Fault Analysis (PhoBERT)
=====================================================================
Pipeline: Vietnamese text → Preprocessing → PhoBERT Tokenization
           → PhoBERT Encoding → Semantic Fault Classification
           → Severity Scoring → Recommendation Generation

Sử dụng PhoBERT (vinai/phobert-base) để encode mô tả thiết bị,
sau đó so sánh cosine similarity với các mẫu lỗi đã định nghĩa
để phân loại lỗi và đánh giá mức độ nghiêm trọng.
"""

import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU mode

import unicodedata
import torch
from dataclasses import dataclass, field
from transformers import AutoModel, AutoTokenizer


# ============================================================
# 1. PhoBERT MODEL LOADER
# ============================================================

print("🔄 Đang tải PhoBERT model (vinai/phobert-base)...")

import pathlib as _pathlib
_MODEL_PATH = str(_pathlib.Path(__file__).resolve().parent / "resources" / "phobert-base")

_tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH)
_model = AutoModel.from_pretrained(_MODEL_PATH)
_model.eval()  # Chế độ inference

# Chọn device
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model.to(_device)

print(f"✅ PhoBERT loaded on {_device}")


# ============================================================
# 2. FAULT REFERENCE DATABASE
# ============================================================

# Các mẫu mô tả lỗi tham chiếu — PhoBERT sẽ so sánh semantic similarity
FAULT_REFERENCES = {
    "Bình thường": {
        "samples": [
            "thiết bị hoạt động bình thường ổn định",
            "máy chạy tốt không có vấn đề gì",
            "hoạt động bình thường không có tiếng ồn lạ",
            "mọi thứ ổn định không phát hiện bất thường",
            "thiết bị vận hành tốt không rung không nóng",
            "máy hoạt động êm không có mùi lạ",
            "tình trạng tốt nhiệt độ bình thường",
            "motor chạy êm ái không có tiếng động lạ",
            "hoạt động bình thường chạy êm ái",
            "máy chạy êm không rung không nóng không mùi",
            "thiết bị chạy ổn định không có bất thường gì",
            "hoạt động tốt không có sự cố",
            "vận hành bình thường không phát hiện hư hỏng",
            "máy hoạt động tốt không cần bảo trì",
            "tất cả chỉ số bình thường thiết bị ổn định",
        ],
        "severity_base": 0.0,
        "is_normal": True,
        "description": "Thiết bị hoạt động bình thường, không phát hiện bất thường",
    },
    "Quá nhiệt": {
        "samples": [
            "thiết bị nóng bất thường nhiệt độ rất cao",
            "motor quá nóng tỏa nhiệt mạnh cháy tay",
            "nhiệt độ tăng cao bất thường bỏng tay nóng ran",
            "thiết bị bốc hơi nóng khói nhiệt tăng",
            "vỏ máy nóng chảy quá nhiệt nghiêm trọng",
        ],
        "severity_base": 0.7,
        "description": "Thiết bị hoạt động ở nhiệt độ cao bất thường",
    },
    "Hỏng bạc đạn / vòng bi": {
        "samples": [
            "rung mạnh kèm tiếng kim loại va chạm",
            "tiếng kêu lạ rung lắc mạnh tiếng cọ sát",
            "rung bất thường tiếng rít cao tiếng kim loại",
            "rung động mạnh kèm tiếng va đập bạc đạn vòng bi",
            "tiếng lách cách rung liên tục giật cục",
        ],
        "severity_base": 0.75,
        "description": "Rung động + tiếng kim loại — nghi ngờ hỏng bạc đạn hoặc vòng bi",
    },
    "Cháy cuộn dây / cháy motor": {
        "samples": [
            "mùi khét cháy kèm nhiệt độ rất cao bốc khói",
            "mùi cháy mùi nhựa cháy nóng bất thường bốc khói",
            "khét mùi dầu cháy quá nhiệt nghiêm trọng khói",
            "motor cháy mùi khét nóng chảy tia lửa điện",
            "cuộn dây cháy mùi cao su cháy bốc khói nhiệt cao",
        ],
        "severity_base": 0.9,
        "description": "Quá nhiệt kết hợp mùi cháy — cháy cuộn dây hoặc cháy motor",
    },
    "Sự cố điện": {
        "samples": [
            "dòng điện tăng đột ngột chập mạch phóng điện",
            "tia lửa điện rò điện chập điện",
            "dòng điện dao động bất thường quá tải cháy cầu chì",
            "mất pha lệch pha sụt áp dòng điện bất thường",
            "điện giật rò điện nguy hiểm chập mạch",
        ],
        "severity_base": 0.85,
        "description": "Vấn đề hệ thống điện — chập, rò, quá tải",
    },
    "Quá tải cơ khí": {
        "samples": [
            "nóng bất thường kèm rung mạnh quá tải",
            "nhiệt độ cao rung động mạnh thiết bị quá tải",
            "quá nóng rung lắc mạnh dây đai căng quá tải",
            "motor nóng rung mạnh chạy chậm công suất giảm",
            "thiết bị quá tải nóng rung giật kẹt",
        ],
        "severity_base": 0.8,
        "description": "Quá nhiệt + rung động — thiết bị bị quá tải",
    },
    "Rò rỉ hệ thống": {
        "samples": [
            "rò rỉ dầu chảy dầu dầu loang",
            "rò rỉ nước xì hơi chảy nước",
            "gioăng hỏng rò rỉ rỉ dầu tràn dầu",
            "phớt hỏng rò rỉ dầu áp suất giảm",
            "seal hỏng rò rỉ nước chảy tràn",
        ],
        "severity_base": 0.5,
        "description": "Rò rỉ dầu, nước, hoặc khí trong hệ thống",
    },
    "Hư hỏng cơ khí": {
        "samples": [
            "nứt vỡ gãy biến dạng cong vênh",
            "mòn nhiều ăn mòn gỉ sét han gỉ",
            "đứt dây đai dây đai mòn tuột",
            "lỏng bu lông lung lay trục lệch trục cong",
            "bạc đạn hỏng vòng bi hỏng mòn nhiều",
        ],
        "severity_base": 0.6,
        "description": "Hư hỏng các bộ phận cơ khí",
    },
    "Âm thanh bất thường": {
        "samples": [
            "tiếng ồn lạ tiếng kêu bất thường",
            "ồn lớn tiếng rít tiếng cọ sát",
            "tiếng va đập tiếng nổ gầm",
            "tiếng kẹt tiếng ù tiếng lạ",
            "kêu to kêu lớn tiếng cạch cạch",
        ],
        "severity_base": 0.5,
        "description": "Phát hiện âm thanh bất thường từ thiết bị",
    },
    "Giảm hiệu suất": {
        "samples": [
            "chạy chậm yếu công suất giảm",
            "không khởi động không chạy dừng đột ngột",
            "tắt đột ngột chập chờn không ổn định",
            "hoạt động chậm kẹt treo máy",
            "quá tốc hiệu suất thấp năng suất giảm",
        ],
        "severity_base": 0.55,
        "description": "Thiết bị hoạt động không đạt hiệu suất mong đợi",
    },
}


# ============================================================
# 3. SYMPTOM KEYWORD DATABASE (for keyword extraction display)
# ============================================================

SYMPTOM_KEYWORDS = {
    "Nhiệt độ": [
        "nóng bất thường", "nóng chảy", "nhiệt độ cao", "nhiệt độ rất cao",
        "quá nhiệt", "quá nóng", "nóng", "nóng ran", "tỏa nhiệt",
        "cháy tay", "bỏng tay", "khói", "bốc khói", "hơi nóng",
    ],
    "Rung động": [
        "rung mạnh", "rung bất thường", "rung lắc", "rung lắc mạnh",
        "rung", "rung nhẹ", "rung liên tục", "dao động mạnh",
        "giật", "giật cục", "lung lay", "xóc",
    ],
    "Âm thanh": [
        "tiếng kim loại va chạm", "tiếng kim loại", "tiếng kêu lạ",
        "tiếng kêu bất thường", "ồn bất thường", "ồn lớn", "tiếng ồn lạ",
        "tiếng rít", "tiếng rít cao", "tiếng cọ sát", "tiếng va đập",
        "tiếng ù", "tiếng lạ", "tiếng kẹt", "tiếng nổ", "nổ",
        "tiếng cạch cạch", "tiếng lách cách", "gầm",
    ],
    "Mùi": [
        "mùi khét", "mùi cháy", "khét", "mùi dầu cháy", "mùi dầu",
        "mùi nhớt cháy", "mùi cao su cháy", "mùi nhựa cháy",
        "mùi lạ", "mùi hôi", "mùi hắc", "bốc mùi", "cháy khét",
    ],
    "Điện": [
        "dòng điện tăng đột ngột", "dòng điện tăng", "dòng điện dao động",
        "dòng điện bất thường", "chập điện", "đánh lửa", "phóng điện",
        "tia lửa", "tia lửa điện", "điện giật", "rò điện", "chập mạch",
        "cháy cầu chì", "quá tải", "sụt áp", "mất pha", "lệch pha",
    ],
    "Rò rỉ": [
        "rò rỉ dầu", "rò rỉ nước", "rò rỉ", "chảy dầu", "chảy nước",
        "rỉ dầu", "dầu loang", "xì", "xì hơi", "tràn dầu",
    ],
    "Cơ khí": [
        "gãy", "nứt", "vỡ", "mòn", "mòn nhiều", "ăn mòn", "gỉ sét",
        "han gỉ", "biến dạng", "cong vênh", "lỏng", "lỏng bu lông",
        "tuột", "đứt dây đai", "dây đai mòn", "bạc đạn hỏng",
        "bạc đạn", "vòng bi", "vòng bi hỏng", "trục bị cong", "trục lệch",
    ],
    "Hiệu suất": [
        "chạy chậm", "yếu", "công suất giảm", "không khởi động",
        "không chạy", "dừng đột ngột", "tắt đột ngột", "chập chờn",
        "không ổn định", "hoạt động chậm", "kẹt", "treo", "quá tốc",
    ],
}

# Vietnamese negation words
NEGATION_WORDS = [
    "không có", "không bị", "không thấy", "không nghe",
    "không phát hiện", "không còn", "không hề",
    "chưa có", "chưa bị", "chưa thấy", "chưa phát hiện",
    "hết", "đã hết", "không", "chưa",
]


# ============================================================
# 4. RECOMMENDATION DATABASE
# ============================================================

RECOMMENDATIONS_DB = {
    "Cháy cuộn dây / cháy motor": [
        "🚨 DỪNG THIẾT BỊ NGAY LẬP TỨC",
        "Ngắt nguồn điện và đảm bảo an toàn khu vực",
        "Kiểm tra cách điện cuộn dây (megger test)",
        "Kiểm tra hệ thống làm mát và quạt gió",
        "Đánh giá lại điều kiện tải — có thể quá tải",
        "Liên hệ kỹ sư điện để kiểm tra chuyên sâu",
    ],
    "Hỏng bạc đạn / vòng bi": [
        "🚨 DỪNG THIẾT BỊ để tránh hư hỏng thêm",
        "Kiểm tra bạc đạn / vòng bi — thay thế nếu cần",
        "Kiểm tra hệ thống bôi trơn — bổ sung mỡ bôi trơn",
        "Kiểm tra độ đồng trục (alignment) các khớp nối",
        "Kiểm tra cân bằng động rotor",
    ],
    "Quá tải cơ khí": [
        "🚨 GIẢM TẢI NGAY hoặc dừng thiết bị",
        "Kiểm tra điều kiện tải hiện tại so với thông số thiết kế",
        "Kiểm tra hệ thống truyền động (dây đai, khớp nối, hộp số)",
        "Kiểm tra hệ thống làm mát",
        "Đánh giá lại quy trình vận hành",
    ],
    "Sự cố điện": [
        "🚨 NGẮT NGUỒN ĐIỆN NGAY",
        "Kiểm tra cách điện toàn bộ hệ thống",
        "Kiểm tra điện áp, dòng điện, hệ số công suất",
        "Kiểm tra tủ điện, CB, contactor, relay bảo vệ",
        "Kiểm tra tiếp địa và hệ thống bảo vệ",
        "Liên hệ kỹ sư điện chuyên trách",
    ],
    "Quá nhiệt": [
        "Giảm tải hoặc dừng thiết bị để hạ nhiệt",
        "Kiểm tra hệ thống làm mát (quạt, nước, dầu)",
        "Kiểm tra điều kiện môi trường (thông gió, nhiệt độ xung quanh)",
        "Kiểm tra hệ thống bôi trơn",
        "Theo dõi nhiệt độ bằng camera nhiệt nếu có",
    ],
    "Rò rỉ hệ thống": [
        "Xác định vị trí rò rỉ chính xác",
        "Kiểm tra gioăng, phớt, seal — thay thế nếu hỏng",
        "Kiểm tra áp suất hệ thống",
        "Bổ sung dầu/nước nếu thiếu",
        "Lên kế hoạch bảo trì thay thế seal",
    ],
    "Hư hỏng cơ khí": [
        "Kiểm tra chi tiết bộ phận bị hư hỏng",
        "Đánh giá mức độ hư hỏng — sửa chữa hoặc thay thế",
        "Kiểm tra các bộ phận liên quan có bị ảnh hưởng không",
        "Lên kế hoạch thay thế phụ tùng",
        "Rà soát lại quy trình bảo trì định kỳ",
    ],
    "Âm thanh bất thường": [
        "Xác định vị trí và đặc điểm âm thanh",
        "Kiểm tra các bộ phận quay: bạc đạn, trục, bánh răng",
        "Kiểm tra hệ thống truyền động (dây đai, xích)",
        "Kiểm tra lỏng kết nối cơ khí",
        "Sử dụng stethoscope công nghiệp để chẩn đoán chính xác",
    ],
    "Giảm hiệu suất": [
        "Kiểm tra điều kiện đầu vào (điện, khí, nước)",
        "Kiểm tra bộ lọc — vệ sinh hoặc thay thế",
        "Kiểm tra hệ thống truyền động — dây đai, ly hợp",
        "Đánh giá lại thông số vận hành",
        "Lên kế hoạch bảo trì tổng thể",
    ],
    "_default": [
        "Tiếp tục theo dõi thiết bị",
        "Bảo dưỡng định kỳ theo kế hoạch",
        "Ghi nhận tình trạng để theo dõi xu hướng",
    ],
}


# ============================================================
# 5. DATA CLASSES
# ============================================================

@dataclass
class AnalysisResult:
    """Kết quả phân tích NLP."""
    fault_type: str
    severity: str
    severity_score: float
    confidence: float
    keywords: list = field(default_factory=list)
    symptoms: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    summary: str = ""
    pipeline_steps: list = field(default_factory=list)


# ============================================================
# 6. NLP ENGINE CLASS (PhoBERT-based)
# ============================================================

class NLPEngine:
    """
    NLP Engine sử dụng PhoBERT cho phân tích thiết bị công nghiệp.

    Pipeline:
    1. Tiền xử lý văn bản (normalize, clean)
    2. PhoBERT Tokenization & Encoding
    3. Cosine Similarity với các mẫu lỗi tham chiếu
    4. Phân loại lỗi (semantic classification)
    5. Trích xuất keyword (supplementary)
    6. Đánh giá severity
    7. Sinh khuyến nghị
    """

    def __init__(self):
        self.tokenizer = _tokenizer
        self.model = _model
        self.device = _device
        self.fault_refs = FAULT_REFERENCES
        self.recommendations_db = RECOMMENDATIONS_DB

        # Pre-compute embeddings cho các mẫu tham chiếu
        self.ref_embeddings = {}
        self._precompute_reference_embeddings()

    def _precompute_reference_embeddings(self):
        """Tính trước embeddings cho tất cả mẫu tham chiếu."""
        print("🔄 Pre-computing reference embeddings...")
        for fault_name, fault_data in self.fault_refs.items():
            embeddings = []
            for sample in fault_data["samples"]:
                emb = self._encode_text(sample)
                embeddings.append(emb)
            # Lấy trung bình các embeddings làm đại diện cho loại lỗi
            self.ref_embeddings[fault_name] = torch.stack(embeddings).mean(dim=0)
        print("✅ Reference embeddings ready")

    # ----------------------------------------------------------
    # PhoBERT Encoding
    # ----------------------------------------------------------
    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text thành embedding vector sử dụng PhoBERT.
        Trả về [CLS] token embedding (768-dim).
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self.device)

        outputs = self.model(**inputs)
        # Lấy [CLS] token (vị trí 0) từ last hidden state
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.squeeze(0).cpu()

    # ----------------------------------------------------------
    # Step 1: Text Preprocessing
    # ----------------------------------------------------------
    def preprocess(self, text: str) -> str:
        """Tiền xử lý văn bản tiếng Việt."""
        text = unicodedata.normalize("NFC", text)
        text = text.lower()
        text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # ----------------------------------------------------------
    # Step 2: Keyword Extraction (supplementary)
    # ----------------------------------------------------------
    def extract_keywords(self, text: str) -> list:
        """
        Trích xuất từ khóa triệu chứng từ text.
        Bổ sung cho PhoBERT — giúp hiển thị keywords cho user.
        """
        found = []
        negated_spans = []

        # Collect all keywords, sort by length desc
        all_kw = []
        for category, keywords in SYMPTOM_KEYWORDS.items():
            for kw in keywords:
                all_kw.append((kw, category))
        all_kw.sort(key=lambda x: len(x[0]), reverse=True)

        matched_spans = []

        for kw, category in all_kw:
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            for match in pattern.finditer(text):
                start, end = match.span()

                # Check if in negated span
                in_negated = any(start >= ns and end <= ne for ns, ne in negated_spans)
                if in_negated:
                    continue

                # Check negation
                lookback = max(0, start - 25)
                preceding = text[lookback:start].strip().lower()
                is_neg = any(preceding.endswith(neg) for neg in NEGATION_WORDS)
                if is_neg:
                    negated_spans.append((start, end))
                    continue

                # Check overlap
                is_overlap = any(start < me and end > ms for ms, me in matched_spans)
                if not is_overlap:
                    matched_spans.append((start, end))
                    found.append({"keyword": kw, "category": category})

        return found

    # ----------------------------------------------------------
    # Step 3: Semantic Fault Classification (PhoBERT)
    # ----------------------------------------------------------
    def classify_fault_phobert(self, text: str) -> list:
        """
        Phân loại lỗi bằng PhoBERT cosine similarity.
        Returns: danh sách (fault_name, similarity_score) đã sắp xếp giảm dần.
        """
        text_embedding = self._encode_text(text)

        scores = []
        for fault_name, ref_emb in self.ref_embeddings.items():
            similarity = torch.nn.functional.cosine_similarity(
                text_embedding.unsqueeze(0),
                ref_emb.unsqueeze(0),
            ).item()
            scores.append((fault_name, similarity))

        # Sắp xếp giảm dần theo similarity
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    # ----------------------------------------------------------
    # Step 4: Severity Assessment
    # ----------------------------------------------------------
    def assess_severity(self, fault_type: str, similarity: float, keywords: list) -> tuple:
        """
        Đánh giá mức độ nghiêm trọng.
        Kết hợp: PhoBERT similarity + severity_base của loại lỗi + số keyword.
        """
        fault_data = self.fault_refs.get(fault_type, {})
        severity_base = fault_data.get("severity_base", 0.3)

        # Kết hợp severity: base * similarity + keyword bonus
        keyword_bonus = min(len(keywords) * 0.05, 0.2)
        severity_score = min(severity_base * similarity + keyword_bonus, 1.0)
        severity_score = round(severity_score, 2)

        if severity_score >= 0.65:
            return ("NGHIÊM TRỌNG", severity_score)
        elif severity_score >= 0.40:
            return ("CẢNH BÁO", severity_score)
        else:
            return ("THẤP", severity_score)

    # ----------------------------------------------------------
    # Step 5: Summary Generation
    # ----------------------------------------------------------
    def generate_summary(self, equipment: str, fault_type: str, severity: str, keywords: list, similarity: float) -> str:
        """Tạo tóm tắt."""
        if fault_type == "Hoạt động ổn định":
            return f"{equipment} — không phát hiện triệu chứng bất thường. Thiết bị có thể đang hoạt động bình thường."

        fault_desc = self.fault_refs.get(fault_type, {}).get("description", fault_type)
        kw_list = ", ".join([k["keyword"] for k in keywords]) if keywords else "không rõ"

        summary = f"{equipment} — PhoBERT phân loại: {fault_type} (similarity: {similarity:.1%}). "
        summary += f"Từ khóa phát hiện: {kw_list}. "
        summary += f"Mô tả: {fault_desc}. Mức độ: {severity}."
        return summary

    # ----------------------------------------------------------
    # MAIN PIPELINE
    # ----------------------------------------------------------
    def analyze(self, equipment: str, description: str) -> AnalysisResult:
        """
        Main NLP pipeline sử dụng PhoBERT.

        Args:
            equipment: Loại thiết bị
            description: Mô tả tự nhiên tiếng Việt

        Returns:
            AnalysisResult
        """
        pipeline_steps = []

        # Step 1: Preprocessing
        cleaned = self.preprocess(description)
        pipeline_steps.append({
            "step": 1,
            "name": "Tiền xử lý văn bản",
            "input": description,
            "output": cleaned,
        })

        # Step 2: PhoBERT Tokenization
        tokens = self.tokenizer.tokenize(cleaned)
        pipeline_steps.append({
            "step": 2,
            "name": "PhoBERT Tokenization",
            "input": cleaned,
            "output": tokens,
        })

        # Step 3: Keyword Extraction
        keywords = self.extract_keywords(cleaned)
        pipeline_steps.append({
            "step": 3,
            "name": "Trích xuất từ khóa",
            "input": cleaned,
            "output": keywords,
        })

        # Step 4: PhoBERT Semantic Classification
        scores = self.classify_fault_phobert(cleaned)
        top_5 = scores[:5]

        pipeline_steps.append({
            "step": 4,
            "name": "PhoBERT Phân loại lỗi (Cosine Similarity)",
            "input": "PhoBERT embedding (768-dim)",
            "output": [{
                "fault": ("✅ " if self.fault_refs.get(f, {}).get("is_normal", False) else "⚠️ ") + f,
                "similarity": round(s, 4),
            } for f, s in top_5],
        })

        # --- KEYWORD-AWARE RE-RANKING ---
        # Kết hợp PhoBERT similarity với keyword detection
        # Nếu có keywords thuộc category nào → boost fault type liên quan
        keyword_categories = set(k["category"] for k in keywords)

        # Mapping category → related fault types
        CATEGORY_FAULT_MAP = {
            "Nhiệt độ": ["Quá nhiệt", "Cháy cuộn dây / cháy motor", "Quá tải cơ khí"],
            "Rung động": ["Hỏng bạc đạn / vòng bi", "Quá tải cơ khí"],
            "Âm thanh": ["Hỏng bạc đạn / vòng bi", "Âm thanh bất thường"],
            "Mùi": ["Cháy cuộn dây / cháy motor"],
            "Điện": ["Sự cố điện", "Cháy cuộn dây / cháy motor"],
            "Rò rỉ": ["Rò rỉ hệ thống"],
            "Cơ khí": ["Hư hỏng cơ khí", "Hỏng bạc đạn / vòng bi"],
            "Hiệu suất": ["Giảm hiệu suất", "Quá tải cơ khí"],
        }

        # Boost scores dựa trên keyword categories
        boosted_scores = []
        for fault_name, sim in scores:
            boost = 0.0
            for cat in keyword_categories:
                related = CATEGORY_FAULT_MAP.get(cat, [])
                if fault_name in related:
                    boost += 0.1  # Boost 0.1 cho mỗi category match
            boosted_scores.append((fault_name, sim + boost))

        boosted_scores.sort(key=lambda x: x[1], reverse=True)
        top_fault, top_score = boosted_scores[0]
        top_sim = dict(scores)[top_fault]  # Original similarity (không boost)

        # --- DECISION LOGIC ---
        # Kiểm tra nếu kết quả là "Bình thường" HOẶC không có keyword nào
        is_normal = self.fault_refs.get(top_fault, {}).get("is_normal", False)

        if is_normal or (len(keywords) == 0):
            fault_type = "Hoạt động ổn định"
            severity = "THẤP"
            severity_score = 0.0
            confidence = round(top_sim, 2)
            recommendations = RECOMMENDATIONS_DB["_default"]
        else:
            fault_type = top_fault
            severity, severity_score = self.assess_severity(fault_type, top_sim, keywords)
            confidence = round(top_sim, 2)
            recommendations = self.recommendations_db.get(fault_type, self.recommendations_db["_default"])

        pipeline_steps.append({
            "step": 5,
            "name": "Đánh giá mức độ nghiêm trọng",
            "input": f"fault={fault_type}, similarity={top_sim:.4f}, keywords={len(keywords)}",
            "output": {"severity": severity, "score": severity_score},
        })

        # Step 6: Recommendations
        pipeline_steps.append({
            "step": 6,
            "name": "Sinh khuyến nghị",
            "input": fault_type,
            "output": recommendations,
        })

        # Summary
        summary = self.generate_summary(equipment, fault_type, severity, keywords, top_sim)

        return AnalysisResult(
            fault_type=fault_type,
            severity=severity,
            severity_score=severity_score,
            confidence=confidence,
            keywords=[k["keyword"] for k in keywords],
            symptoms=[{
                "keyword": k["keyword"],
                "category": k["category"],
                "label": k["keyword"],
                "weight": 3,
            } for k in keywords],
            recommendations=recommendations,
            summary=summary,
            pipeline_steps=pipeline_steps,
        )


# ============================================================
# SINGLETON & CONVENIENCE
# ============================================================

engine = NLPEngine()


def analyze(equipment: str, description: str) -> AnalysisResult:
    """Convenience function — gọi engine.analyze()."""
    return engine.analyze(equipment, description)
