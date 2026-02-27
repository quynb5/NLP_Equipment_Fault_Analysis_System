"""
TF-IDF Engine — Vietnamese Industrial Equipment Fault Analysis
================================================================
Sử dụng TF-IDF + Logistic Regression/SVM (pre-trained) để phân loại lỗi.
Lightweight & fast inference so với PhoBERT.

Pipeline:
  Input text → TF-IDF vectorize → Classifier predict → AnalysisResult
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from backend.core.base_engine import BaseNLPEngine, AnalysisResult


# Model directory
_MODEL_DIR = Path(__file__).resolve().parent.parent / "resources" / "tfidf"


# ============================================================
# SEVERITY & RECOMMENDATION MAPPINGS
# ============================================================

SEVERITY_MAP = {
    "Hoạt động ổn định": ("BÌNH THƯỜNG", 0.0),
    "Quá nhiệt": ("NGHIÊM TRỌNG", 0.7),
    "Hỏng bạc đạn / vòng bi": ("NGHIÊM TRỌNG", 0.75),
    "Cháy cuộn dây / cháy motor": ("NGHIÊM TRỌNG", 0.9),
    "Sự cố điện": ("NGHIÊM TRỌNG", 0.85),
    "Quá tải cơ khí": ("CẢNH BÁO", 0.8),
    "Rò rỉ hệ thống": ("CẢNH BÁO", 0.5),
    "Hư hỏng cơ khí": ("CẢNH BÁO", 0.6),
    "Âm thanh bất thường": ("CẢNH BÁO", 0.5),
    "Giảm hiệu suất": ("CẢNH BÁO", 0.55),
}

RECOMMENDATIONS_MAP = {
    "Hoạt động ổn định": [
        "Tiếp tục vận hành bình thường",
        "Lên lịch bảo trì định kỳ theo kế hoạch",
    ],
    "Quá nhiệt": [
        "Dừng thiết bị ngay, để nguội trước khi kiểm tra",
        "Kiểm tra hệ thống làm mát, quạt tản nhiệt",
        "Kiểm tra dầu bôi trơn, tra dầu nếu thiếu",
    ],
    "Hỏng bạc đạn / vòng bi": [
        "Dừng máy ngay để tránh hư hỏng lan rộng",
        "Thay thế bạc đạn / vòng bi bị hỏng",
        "Kiểm tra trục máy, cân chỉnh lại",
    ],
    "Cháy cuộn dây / cháy motor": [
        "NGẮT ĐIỆN NGAY LẬP TỨC",
        "Không cố khởi động lại, chờ kiểm tra chuyên gia",
        "Kiểm tra cách điện cuộn dây, thay motor nếu cần",
    ],
    "Sự cố điện": [
        "Ngắt nguồn điện ngay, kiểm tra an toàn",
        "Kiểm tra cầu chì, relay bảo vệ",
        "Kiểm tra dây dẫn, đầu nối, tiếp xúc điện",
    ],
    "Quá tải cơ khí": [
        "Giảm tải ngay cho thiết bị",
        "Kiểm tra dây đai, khớp nối truyền động",
        "Xác định nguyên nhân quá tải, điều chỉnh công suất",
    ],
    "Rò rỉ hệ thống": [
        "Xác định vị trí rò rỉ chính xác",
        "Thay thế gioăng, phớt, seal bị hỏng",
        "Kiểm tra áp suất hệ thống sau sửa chữa",
    ],
    "Hư hỏng cơ khí": [
        "Dừng máy, đánh giá mức độ hư hỏng",
        "Thay thế chi tiết bị hỏng (bu lông, dây đai, trục)",
        "Kiểm tra cân bằng, lắp đặt lại thiết bị",
    ],
    "Âm thanh bất thường": [
        "Giảm tải hoặc dừng máy để kiểm tra",
        "Xác định nguồn phát ra âm thanh lạ",
        "Kiểm tra bạc đạn, bánh răng, dây đai",
    ],
    "Giảm hiệu suất": [
        "Kiểm tra nguồn điện cung cấp",
        "Kiểm tra tải, giảm tải nếu quá mức",
        "Kiểm tra bộ điều khiển, sensor, relay",
    ],
}


class TFIDFEngine(BaseNLPEngine):
    """
    TF-IDF + Classical ML Engine cho phân tích thiết bị công nghiệp.
    Lightweight, fast inference (~1-2ms so với ~30-50ms PhoBERT).
    """

    @property
    def name(self) -> str:
        return "tfidf"

    def __init__(self):
        """Load pre-trained vectorizer, classifier, label encoder."""
        print("🔄 Loading TF-IDF model artifacts...")

        self.vectorizer = joblib.load(_MODEL_DIR / "vectorizer.pkl")
        self.classifier = joblib.load(_MODEL_DIR / "classifier.pkl")
        self.label_encoder = joblib.load(_MODEL_DIR / "label_encoder.pkl")

        # Load metadata for version info
        meta_path = _MODEL_DIR / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        print(f"✅ TF-IDF model loaded (v{self.metadata.get('version', '?')})")
        print(f"   Classifier: {self.metadata.get('classifier', '?')}")
        print(f"   Classes: {len(self.label_encoder.classes_)}")

    def analyze(self, equipment: str, description: str) -> AnalysisResult:
        """
        Phân tích thiết bị bằng TF-IDF pipeline.

        Args:
            equipment: Loại thiết bị
            description: Mô tả tình trạng thiết bị

        Returns:
            AnalysisResult
        """
        t0 = time.perf_counter()
        pipeline_steps = []

        # Step 1: TF-IDF vectorize
        X = self.vectorizer.transform([description])
        pipeline_steps.append({
            "step": 1,
            "name": "TF-IDF Vectorization",
            "output": f"Sparse vector: {X.shape}",
        })

        # Step 2: Predict class
        y_pred = self.classifier.predict(X)[0]
        fault_type = self.label_encoder.inverse_transform([y_pred])[0]

        # Step 3: Confidence score
        if hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(X)[0]
            confidence = float(np.max(proba))
        elif hasattr(self.classifier, "decision_function"):
            decision = self.classifier.decision_function(X)[0]
            # Convert decision function to pseudo-probability via softmax
            exp_d = np.exp(decision - np.max(decision))
            proba = exp_d / exp_d.sum()
            confidence = float(np.max(proba))
        else:
            confidence = 0.0

        pipeline_steps.append({
            "step": 2,
            "name": "Classification",
            "output": f"{fault_type} (confidence: {confidence:.4f})",
        })

        # Step 4: Severity
        severity, severity_score = SEVERITY_MAP.get(
            fault_type, ("CẢNH BÁO", 0.5)
        )

        # Adjust severity_score by confidence
        severity_score = round(severity_score * confidence, 3)

        pipeline_steps.append({
            "step": 3,
            "name": "Severity Assessment",
            "output": f"{severity} (score: {severity_score})",
        })

        # Step 5: Extract keywords from TF-IDF features
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_scores = X.toarray()[0]
        top_indices = tfidf_scores.argsort()[-5:][::-1]
        keywords = [
            feature_names[i] for i in top_indices if tfidf_scores[i] > 0
        ]

        pipeline_steps.append({
            "step": 4,
            "name": "Keyword Extraction (TF-IDF top features)",
            "output": keywords,
        })

        # Step 6: Recommendations
        recommendations = RECOMMENDATIONS_MAP.get(fault_type, [
            "Kiểm tra thiết bị",
            "Liên hệ kỹ thuật viên",
        ])

        # Step 7: Summary
        summary = (
            f"[TF-IDF] Thiết bị '{equipment}': {fault_type}. "
            f"Mức độ: {severity} ({severity_score}). "
            f"Từ khóa: {', '.join(keywords[:3])}."
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return AnalysisResult(
            fault_type=fault_type,
            severity=severity,
            severity_score=severity_score,
            confidence=round(confidence, 4),
            keywords=keywords,
            symptoms=[{
                "keyword": k,
                "category": "tfidf_feature",
                "label": k,
                "weight": 3,
            } for k in keywords],
            recommendations=recommendations,
            summary=summary,
            pipeline_steps=pipeline_steps,
            engine_name=self.name,
            engine_latency_ms=round(elapsed_ms, 2),
        )
