"""
PhoBERT Engine - Vietnamese Industrial Equipment Fault Analysis
===============================================================
Pipeline: Vietnamese text → Preprocessing → PhoBERT Tokenization
           → PhoBERT Encoding → Fault Classification
           → Severity Scoring → Recommendation Generation

Hỗ trợ 2 chế độ:
  1. Fine-tuned Classifier (nếu có model đã train)
  2. Zero-shot Cosine Similarity (fallback)
"""

import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU mode

import time
import unicodedata
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from backend.core.base_engine import BaseNLPEngine, AnalysisResult


# ============================================================
# 1. PhoBERT MODEL LOADER
# ============================================================

_HF_MODEL_NAME = "vinai/phobert-base"

import pathlib as _pathlib
_MODEL_PATH = str(_pathlib.Path(__file__).resolve().parent.parent / "resources" / "phobert-base")


def _load_model():
    """Load PhoBERT từ local path, nếu không có thì download từ HuggingFace."""
    model_dir = _pathlib.Path(_MODEL_PATH)

    if model_dir.exists() and any(model_dir.iterdir()):
        # Load từ local path
        print(f"🔄 Đang tải PhoBERT từ local: {_MODEL_PATH}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH)
            model = AutoModel.from_pretrained(_MODEL_PATH)
            print("✅ Load từ local thành công")
            return tokenizer, model
        except Exception as e:
            print(f"⚠️ Load từ local thất bại: {e}")
            print("🔄 Sẽ download lại từ HuggingFace...")

    # Download từ HuggingFace và lưu vào _MODEL_PATH
    print(f"🔄 Đang download PhoBERT ({_HF_MODEL_NAME}) từ HuggingFace...")
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(_HF_MODEL_NAME)
    model = AutoModel.from_pretrained(_HF_MODEL_NAME)
    tokenizer.save_pretrained(_MODEL_PATH)
    model.save_pretrained(_MODEL_PATH)
    print(f"✅ Download và lưu thành công tại: {_MODEL_PATH}")
    return tokenizer, model


_tokenizer, _model = _load_model()
_model.eval()  # Chế độ inference

# Chọn device
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model.to(_device)

print(f"✅ PhoBERT ready on {_device}")


# ============================================================
# 1b. FINE-TUNED CLASSIFIER (optional)
# ============================================================

class _PhoBERTClassifier(nn.Module):
    """PhoBERT + Linear classification head."""
    def __init__(self, phobert_model, num_classes=10, dropout=0.3):
        super().__init__()
        self.phobert = phobert_model
        hidden_size = self.phobert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


_FINETUNED_DIR = _pathlib.Path(__file__).resolve().parent.parent / "resources" / "phobert-finetuned"
_finetuned_model = None
_finetuned_labels = None


def _load_finetuned_model():
    """Load fine-tuned PhoBERT classifier nếu có."""
    global _finetuned_model, _finetuned_labels

    model_path = _FINETUNED_DIR / "model.pt"
    head_path = _FINETUNED_DIR / "classifier_head.pt"

    if not model_path.exists():
        print("ℹ️  Fine-tuned model not found → using zero-shot similarity")
        return

    try:
        head_info = torch.load(head_path, map_location="cpu", weights_only=False)
        num_classes = head_info["num_classes"]
        dropout_p = head_info.get("dropout_p", 0.3)
        _finetuned_labels = head_info["label_classes"]

        classifier = _PhoBERTClassifier(
            phobert_model=_model,
            num_classes=num_classes,
            dropout=dropout_p,
        )

        state_dict = torch.load(model_path, map_location=_device, weights_only=False)
        classifier.load_state_dict(state_dict, strict=False)
        classifier.to(_device)
        classifier.eval()

        _finetuned_model = classifier
        print(f"✅ Fine-tuned PhoBERT classifier loaded ({num_classes} classes)")
        print(f"   Labels: {_finetuned_labels}")
    except Exception as e:
        print(f"⚠️  Failed to load fine-tuned model: {e}")
        print("   Falling back to zero-shot similarity")


_load_finetuned_model()


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
            # --- Negation & new patterns ---
            "motor vận hành trơn tru không có tiếng ồn bất thường nhiệt độ trong ngưỡng",
            "bơm nước chạy đều áp suất ổn định không rung không nóng",
            "quạt công nghiệp hoạt động bình thường không phát hiện dấu hiệu bất thường",
            "hệ thống bơm thủy lực vận hành ổn định suốt ca không rò rỉ không tiếng kêu",
            "máy nén khí chạy tốt không nóng bất thường công suất đạt yêu cầu",
            "thiết bị không có vấn đề gì thông số nằm trong giới hạn an toàn",
            "motor chạy mượt mà không rung lắc không mùi khét nhiệt độ bình thường",
            "băng tải vận hành đúng tốc độ thiết kế không trượt không kẹt",
            "hệ thống chạy êm nhiệt độ ổn định không rung không có bất thường",
            "thiết bị chạy ổn định suốt ca làm việc không có bất thường gì",
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
            "máy nóng ran sờ vào bỏng da nhiệt cao",
            "nhiệt tỏa ra rất mạnh thiết bị quá nóng",
            "vỏ motor nóng dữ dội nhiệt độ vượt ngưỡng",
            "bơm thủy lực nóng bất thường chạm bỏng tay",
            "nóng hầm hập nhiệt lượng tỏa lớn bất thường",
            "nhiệt tăng liên tục không giảm nóng bốc hơi",
            "cảm biến nhiệt báo vượt ngưỡng thiết bị quá nóng",
            "nhiệt độ cao bất thường nóng hơn bình thường rất nhiều",
            "thiết bị phát nhiệt mạnh quá mức cho phép",
            "motor nóng liên tục không hạ nhiệt dù giảm tải",
            # --- New patterns ---
            "vỏ motor nóng hơn bình thường rất nhiều khi chạm vào",
            "nhiệt độ bề mặt thiết bị vượt ngưỡng cảnh báo sensor báo quá nhiệt",
            "motor phát nhiệt liên tục dù tải nhẹ hệ thống làm mát không hiệu quả",
            "bơm nóng ran không chạm được nhiệt kế đo vượt 90 độ",
            "nhiệt tỏa ra từ thiết bị lớn bất thường dầu bôi trơn bị loãng",
            "thiết bị nóng bức xạ quạt tản nhiệt chạy hết công suất không giảm nhiệt",
            "thân máy nóng hơn mức cho phép cảm biến nhiệt liên tục cảnh báo",
            "máy nén nóng quá hơi nóng bốc lên mạnh nhiệt tăng bất thường",
            "motor quá nóng nóng bất thường so với mọi khi nhiệt cao",
            "máy bốc hơi nóng nhiệt tăng liên tục không giảm quá nhiệt",
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
            "rung lắc kèm tiếng cọ kim loại bên trong motor",
            "tiếng kêu rít từ vòng bi rung lắc mạnh",
            "trục máy rung giật cục tiếng kêu lạ liên tục",
            "tiếng kim loại cọ sát mạnh rung giật liên tục",
            "bạc đạn phát tiếng kêu rít rung lắc tăng dần",
            "ổ bi kêu to rung lắc rõ rệt khi vận hành",
            "tiếng cọ sát kim loại cảm nhận rung mạnh ở trục",
            "motor giật cục liên tục tiếng lách cách bạc đạn",
            "rung dữ dội kèm âm thanh kim loại va chạm ổ bi",
            "phát hiện rung bất thường tiếng rít cao từ vòng bi",
            # --- New patterns (negation + technical) ---
            "motor không nóng nhưng rung lắc mạnh phát tiếng kêu kim loại",
            "nghe tiếng rít từ vị trí bạc đạn motor rung tăng dần theo thời gian",
            "vòng bi phía sau motor có dấu hiệu mòn tiếng kêu rè khi quay",
            "trục motor lỏng lẻo lắc qua lại khi quay nghi ngờ bạc đạn mòn",
            "rung tần số cao phát ra từ ổ trục dầu mỡ bôi trơn bị biến màu",
            "motor rung kèm tiếng lạch cạch nhịp đều đặn khi quay chậm",
            "ổ bi phát ra tiếng ù liên tục bề mặt trục bị xước",
            "không nóng không khét chỉ rung mạnh kèm tiếng kim loại ổ bi",
            "motor rung lắc bất thường ổ trục phát tiếng kêu nghi bạc đạn hỏng",
            "tiếng rít cao từ bạc đạn kèm rung khi tải tăng",
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
            "bốc khói đen mùi cháy khét nồng nặc từ motor",
            "mùi khét cháy rất nặng kèm khói bốc mạnh",
            "khói bốc ra từ cuộn dây motor mùi cháy nồng",
            "motor phát mùi khét dữ dội bốc khói nóng chảy",
            "phát hiện khói mùi cháy tia lửa từ cuộn dây",
            "mùi dầu cháy khét từ motor bốc khói đen nóng cao",
            "motor bốc khói mùi cháy nhựa nóng không chạm được",
            "cuộn dây motor cháy đen mùi khét bốc khói liên tục",
            "thiết bị bốc khói mùi khét cháy cực kỳ nóng",
            "mùi cháy khét nồng nặc motor nóng chảy bốc khói",
            # --- New patterns (negation + technical) ---
            "thiết bị không rung nhưng bốc mùi khét cháy dữ dội kèm khói đen",
            "cuộn dây stator bị ngắn mạch bốc khói mùi nhựa nóng chảy",
            "motor phát tia lửa bên trong mùi cháy khét nồng nặc lan khắp phòng",
            "lớp cách điện cuộn dây bị chảy motor bốc khói trắng dừng quay",
            "phát hiện váng cháy đen trên cuộn dây khi tháo motor kiểm tra",
            "motor bất ngờ dừng kèm mùi cháy kiểm tra thấy dây quấn đứt cháy",
            "không rung không ồn nhưng mùi khét dữ dội bốc khói từ motor",
            "cuộn dây motor bị đoản mạch cháy đen mùi khét bốc khói nặng",
            "motor cháy cuộn dây stator bị ngắn mạch chập giữa các pha",
            "mùi cháy khét nặng nề motor dừng đột ngột bốc khói đen",
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
            "chập điện cháy cầu chì sụt áp nghiêm trọng",
            "rò điện ra vỏ thiết bị chạm vào bị giật",
            "chập mạch tủ điện điều khiển cầu chì nổ",
            "tia lửa điện phóng ra từ đầu nối nguy hiểm",
            "dòng điện tăng đột ngột vượt giới hạn cho phép",
            "sụt áp trầm trọng thiết bị chạy yếu điện không ổn",
            "lệch pha gây rung motor dòng điện bất thường",
            "rò điện nghiêm trọng vỏ máy điện giật",
            "relay bảo vệ nhảy ngắt liên tục chập điện",
            "dòng điện quá tải gây cháy cầu chì bảo vệ",
            # --- New patterns (negation + technical terms) ---
            "thiết bị không nóng không rung nhưng dòng điện dao động bất thường liên tục",
            "MCB bảo vệ nhảy liên tục khi khởi động motor nghi ngờ chạm mass",
            "dây cáp nguồn bị chảy vỏ nhựa do quá dòng đầu nối bị đen oxy hóa",
            "biến tần báo lỗi quá dòng pha R motor khởi động rồi tự ngắt",
            "aptomat chống rò nhảy khi motor chạy nghi ngờ rò điện ra thân vỏ",
            "motor bị mất pha gây rung bất thường cầu chì một pha bị đứt",
            "relay nhiệt tác động ngắt motor liên tục dù tải không đổi",
            "đầu nối cáp bị hồ quang điện mùi ozone tia lửa phóng ra",
            "không nóng không rung chỉ có dòng điện bất ổn MCB nhảy liên tục",
            "biến tần báo lỗi sự cố điện motor không khởi động được",
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
            "motor nóng rung mạnh chạy yếu quá tải liên tục",
            "thiết bị quá tải nóng kèm rung lắc mạnh giật cục",
            "nóng bất thường kèm rung do chạy quá công suất",
            "nóng rung mạnh dây đai căng quá mức do quá tải",
            "quá tải liên tục gây nóng motor rung mạnh giảm công suất",
            "máy nóng chạy chậm hẳn do quá tải cơ khí nghiêm trọng",
            "rung mạnh kèm nóng thiết bị bị kẹt do quá tải",
            "motor nóng ran rung lắc tải vượt thông số thiết kế",
            "máy nén quá tải nóng rung mạnh công suất sụt giảm",
            "tải quá nặng khiến motor nóng rung chạy chậm lại",
            # --- New patterns (negation + cross-symptom) ---
            "motor không khét không cháy nhưng nóng kèm rung do chạy vượt công suất",
            "thiết bị kẹt tải nặng ampe kế chỉ vượt định mức motor rít chạy chậm",
            "băng tải bị quá tải do hàng hóa chất nhiều gây motor căng dây đai",
            "motor kéo tải quá nặng dây đai trượt liên tục phát mùi cao su",
            "máy bơm bị quá tải do van đầu ra đóng áp suất tăng motor rung giật",
            "công suất yêu cầu vượt xa thông số thiết kế motor nóng kèm tiếng rít",
            "tải cơ khí quá lớn khiến motor chạy chậm hẳn dòng điện tăng gấp đôi",
            "không cháy không khét chỉ nóng rung do kéo tải vượt công suất máy",
            "quá tải làm motor nóng rung mạnh ampe tăng cao dây đai trượt",
            "thiết bị kẹt do tải quá lớn nóng rung giật motor chạy chậm",
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
            "dầu chảy tràn ra nền gioăng bị rách hỏng nặng",
            "rò rỉ dầu thủy lực từ ống nối dầu loang khắp nơi",
            "phớt bơm hỏng gây rò rỉ nước liên tục",
            "xì hơi từ van áp suất giảm do rò rỉ khí",
            "seal bị mòn gây rỉ dầu từ trục bơm",
            "rò rỉ nước làm mát từ đường ống chảy nước liên tục",
            "dầu rò rỉ từ hộp số vết dầu loang trên sàn",
            "bơm thủy lực rò dầu từ phớt trục áp suất sụt",
            "rỉ dầu nhỏ giọt liên tục từ đáy thiết bị",
            "xì khí từ đường ống áp suất cao rò rỉ nghiêm trọng",
            # --- New patterns (negation + technical) ---
            "không nóng không rung nhưng phát hiện vũng dầu dưới đáy máy nén",
            "ống dẫn dầu thủy lực bị rạn nứt gây rò rỉ nhỏ giọt liên tục",
            "van xả an toàn bị xì hơi liên tục áp suất bình chứa giảm dần",
            "mặt bích đường ống bị chảy nước tại vị trí gioăng lắp đặt",
            "cylinder thủy lực bị rỉ dầu từ vị trí phớt trượt hành trình không đủ",
            "hệ thống khí nén bị xì hơi nhiều vị trí compressor chạy liên tục bù áp",
            "bể chứa dầu giảm mức báo động dù không sử dụng nghi rò rỉ đường ống",
            "không ồn không nóng chỉ phát hiện vết dầu rò rỉ dưới máy",
            "phớt trục chính bị hỏng gây rò dầu thủy lực nặng",
            "rò rỉ khí nén từ khớp nối ống áp suất giảm dần liên tục",
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
            "trục bị cong vênh bu lông lỏng nhiều chỗ",
            "dây đai bị đứt rách không truyền động được",
            "bánh răng bị mòn nhiều ăn khớp không đều",
            "gỉ sét nặng bề mặt kim loại han gỉ ăn mòn",
            "thân máy bị nứt vỡ biến dạng nghiêm trọng",
            "bu lông cố định bị lỏng thiết bị lung lay",
            "trục truyền động bị cong lệch tâm nghiêm trọng",
            "chi tiết máy bị gãy nứt do mỏi vật liệu",
            "khớp nối bị mòn xước ăn mòn bề mặt nặng",
            "vỏ thiết bị bị nứt vỡ do va đập cơ khí",
            # --- New patterns (negation + technical) ---
            "không ồn không nóng nhưng kiểm tra thấy nứt thân vỏ máy bơm",
            "bánh răng hộp số bị sứt mẻ nhiều răng chạy giật cục khi vào tải",
            "chốt khớp nối giữa motor và bơm bị cắt đứt do mỏi kim loại",
            "cánh quạt bị nứt gãy mất cân bằng gây rung khi chạy tốc độ cao",
            "trục truyền động bị xoắn biến dạng sau sự cố kẹt tải đột ngột",
            "đế lắp thiết bị bị nứt foundation bu lông neo bị nhổ gây lệch trục",
            "khớp nối mềm bị rách vỡ cao su đệm không truyền lực hiệu quả",
            "không nóng không rung nhưng phát hiện nứt vỡ cơ khí khi kiểm tra",
            "bu lông cố định lỏng thiết bị lung lay khi chạy hư hỏng cơ khí",
            "phát hiện gãy nứt chi tiết máy do mỏi vật liệu sau kiểm tra",
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
            "tiếng ồn bất thường kêu to liên tục từ motor",
            "tiếng rít cao phát ra từ thiết bị khi vận hành",
            "tiếng va đập lớn bên trong máy ồn bất thường",
            "tiếng ù to liên tục từ motor chính",
            "thiết bị phát tiếng kẹt nghiến khi hoạt động",
            "âm thanh lạ phát ra liên tục ồn hơn bình thường",
            "tiếng nổ nhỏ lách tách liên tục từ thiết bị",
            "phát ra tiếng cọ sát lạ khi motor quay",
            "tiếng ồn lớn bất thường khi thiết bị khởi động",
            "tiếng kêu bất thường nghe rõ từ xa máy chạy",
            # --- New patterns (negation + varied) ---
            "thiết bị không nóng không rung nhưng phát ra tiếng ù vù liên tục lạ",
            "nghe tiếng tách tách đều đặn bên trong hộp số khi motor chạy",
            "motor phát ra tiếng huýt sáo cao tần khi tăng tốc bất thường",
            "tiếng gõ lớn nhịp đều phát ra từ đầu piston máy nén mỗi khi nén",
            "quạt hút phát ra tiếng rung rền lạ khi tốc độ gió thay đổi",
            "tiếng lạch xạch liên tục từ bên trong motor dù không tải",
            "âm vang bất thường khi chạy không tải tắt máy thì hết tiếng",
            "không nóng không khét nhưng tiếng ồn bất thường phát ra từ motor",
            "tiếng rền lạ phát ra từ thiết bị khi vận hành âm thanh bất thường",
            "thiết bị phát tiếng ồn lạ nghe rõ khi chạy không tải",
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
            "máy chập chờn tắt bật liên tục không ổn định",
            "thiết bị chạy chậm hẳn công suất sụt giảm rõ rệt",
            "motor không khởi động được bấm nút không phản hồi",
            "máy dừng đột ngột giữa chừng khi đang vận hành",
            "thiết bị hoạt động yếu không đạt công suất thiết kế",
            "motor chạy chậm hơn bình thường năng suất giảm",
            "thiết bị tắt đột ngột không khởi động lại được",
            "hiệu suất giảm rõ rệt máy chạy yếu hẳn",
            "thiết bị không đạt tốc độ yêu cầu chạy chậm",
            "công suất đầu ra giảm mạnh so với thông số thiết kế",
            # --- New patterns (negation + operational) ---
            "không ồn không nóng nhưng máy chạy ngày càng chậm công suất sụt rõ rệt",
            "bơm bơm không đủ lưu lượng dù motor chạy đủ vòng tua nghi cánh bơm mòn",
            "motor khởi động lâu hơn bình thường mất gần 30 giây mới đạt vòng quay",
            "thiết bị tự ngắt giữa chừng rồi khởi động lại liên tục không ổn định",
            "năng suất sản xuất giảm 30 phần trăm so với tháng trước dù cùng tải",
            "quạt quay chậm hơn bình thường dù điện áp cung cấp đúng thông số",
            "motor chạy nhưng moment xoắn yếu không đủ kéo tải như thiết kế",
            "không nóng không rung nhưng công suất đầu ra giảm rõ rệt hiệu suất thấp",
            "thiết bị chạy yếu hẳn hiệu suất giảm dần theo thời gian",
            "máy hoạt động nhưng năng suất thấp không đạt yêu cầu sản xuất",
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


# AnalysisResult imported from base_engine.py


# ============================================================
# 6. PhoBERT ENGINE CLASS
# ============================================================

class PhoBERTEngine(BaseNLPEngine):
    """
    PhoBERT Engine cho phân tích thiết bị công nghiệp.

    Pipeline:
    1. Tiền xử lý văn bản (normalize, clean)
    2. PhoBERT Tokenization & Encoding
    3. Cosine Similarity với các mẫu lỗi tham chiếu
    4. Phân loại lỗi (semantic classification)
    5. Trích xuất keyword (supplementary)
    6. Đánh giá severity
    7. Sinh khuyến nghị
    """

    @property
    def name(self) -> str:
        return "phobert"

    def __init__(self):
        self.tokenizer = _tokenizer
        self.model = _model
        self.device = _device
        self.fault_refs = FAULT_REFERENCES
        self.recommendations_db = RECOMMENDATIONS_DB

        # Fine-tuned classifier (nếu có)
        self.finetuned_model = _finetuned_model
        self.finetuned_labels = _finetuned_labels
        self.use_finetuned = _finetuned_model is not None

        if self.use_finetuned:
            print("🔥 PhoBERTEngine: using FINE-TUNED classifier")
        else:
            print("🔄 PhoBERTEngine: using zero-shot similarity (fallback)")

        # Pre-compute embeddings (luôn cần cho severity assessment)
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
        Phân loại lỗi bằng PhoBERT.
        - Nếu có fine-tuned model → softmax classifier
        - Nếu không → fallback cosine similarity
        Returns: danh sách (fault_name, score) đã sắp xếp giảm dần.
        """
        if self.use_finetuned:
            return self._classify_finetuned(text)
        else:
            return self._classify_zero_shot(text)

    @torch.no_grad()
    def _classify_finetuned(self, text: str) -> list:
        """Phân loại bằng fine-tuned classifier (softmax)."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self.device)

        logits = self.finetuned_model(**inputs)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

        scores = []
        for idx, label in enumerate(self.finetuned_labels):
            scores.append((label, probs[idx].item()))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _classify_zero_shot(self, text: str) -> list:
        """Fallback: phân loại bằng cosine similarity."""
        text_embedding = self._encode_text(text)

        scores = []
        for fault_name, ref_emb in self.ref_embeddings.items():
            similarity = torch.nn.functional.cosine_similarity(
                text_embedding.unsqueeze(0),
                ref_emb.unsqueeze(0),
            ).item()
            scores.append((fault_name, similarity))

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
        t0 = time.perf_counter()
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

        # Step 4: PhoBERT Classification
        classify_method = "Fine-tuned Classifier" if self.use_finetuned else "Cosine Similarity"
        scores = self.classify_fault_phobert(cleaned)
        top_5 = scores[:5]

        pipeline_steps.append({
            "step": 4,
            "name": f"PhoBERT Phân loại lỗi ({classify_method})",
            "input": "PhoBERT embedding (768-dim)",
            "output": [{
                "fault": ("✅ " if self.fault_refs.get(f, {}).get("is_normal", False) else "⚠️ ") + f,
                "score": round(s, 4),
            } for f, s in top_5],
        })

        # --- DECISION LOGIC ---
        if self.use_finetuned:
            # Fine-tuned classifier: scores đã là softmax probabilities, dùng trực tiếp
            top_fault, top_score = scores[0]
            top_sim = top_score  # Probability thay cho similarity

            # Map label name nếu cần
            is_normal = (top_fault == "Hoạt động ổn định")

            if is_normal:
                fault_type = "Hoạt động ổn định"
                severity = "BÌNH THƯỜNG"
                severity_score = 0.0
                confidence = round(top_sim, 2)
                recommendations = RECOMMENDATIONS_DB["_default"]
            else:
                fault_type = top_fault
                severity, severity_score = self.assess_severity(fault_type, top_sim, keywords)
                confidence = round(top_sim, 2)
                recommendations = self.recommendations_db.get(fault_type, self.recommendations_db["_default"])
        else:
            # Zero-shot mode: áp dụng keyword re-ranking + Bình thường heuristics
            keyword_categories = set(k["category"] for k in keywords)

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

            boosted_scores = []
            for fault_name, sim in scores:
                boost = 0.0
                for cat in keyword_categories:
                    related = CATEGORY_FAULT_MAP.get(cat, [])
                    if fault_name in related:
                        boost += 0.1
                boosted_scores.append((fault_name, sim + boost))

            boosted_scores.sort(key=lambda x: x[1], reverse=True)
            top_fault, top_score = boosted_scores[0]
            top_sim = dict(scores)[top_fault]

            is_normal = self.fault_refs.get(top_fault, {}).get("is_normal", False)

            if is_normal and len(keywords) > 0:
                for fname, fscore in boosted_scores:
                    if not self.fault_refs.get(fname, {}).get("is_normal", False):
                        top_fault = fname
                        top_score = fscore
                        top_sim = dict(scores)[fname]
                        break
                is_normal = False

            if is_normal and len(keywords) == 0:
                fault_type = "Hoạt động ổn định"
                severity = "BÌNH THƯỜNG"
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

        elapsed_ms = (time.perf_counter() - t0) * 1000

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
            engine_name=self.name,
            engine_latency_ms=round(elapsed_ms, 2),
        )


# ============================================================
# SINGLETON & CONVENIENCE
# ============================================================

engine = PhoBERTEngine()


def analyze(equipment: str, description: str) -> AnalysisResult:
    """Convenience function — gọi engine.analyze()."""
    return engine.analyze(equipment, description)
