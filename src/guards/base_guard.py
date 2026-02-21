import asyncio
import os
from dotenv import load_dotenv
from groq import AsyncGroq
import fasttext
import unicodedata
import ahocorasick
from pathlib import Path
from src.guards.prompts import prompts  
import logging
logger = logging.getLogger(__name__)
# load_dotenv()

class BaseGuard:
    def __init__(
        self,
        guard_model: str,
        fasttext_model_dir: str,
        blocked_file_path: str,
        
    ):
        self.groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = guard_model  # hoặc model Groq khác nếu muốn

        # FastText language detection
        self.language_model = fasttext.load_model(str(fasttext_model_dir))
        self.max_chars = 4000
        self.lang_threshold = 0.80
        self.allowed_lang = "__label__vi"

        # Aho-Corasick blocked keywords
        BLOCKED_FILE = blocked_file_path

        blocked_list = []
        if BLOCKED_FILE.is_file():
            try:
                content = BLOCKED_FILE.read_text(encoding="utf-8")
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith(("#", "//")):
                        blocked_list.append(line)
                logger.info(f"[BaseGuard] Loaded {len(blocked_list)} blocked keywords from {BLOCKED_FILE}")
            except Exception as e:
                logger.error(f"[ERROR] Không đọc được file blocked keywords: {e}")
        else:
            logger.warning(f"[WARNING] Không tìm thấy file blocked keywords: {BLOCKED_FILE}. Sử dụng danh sách rỗng.")
        
        self.blocker = ahocorasick.Automaton()
        for keyword in blocked_list:
            key = keyword.lower()
            self.blocker.add_word(key, keyword)
        self.blocker.make_automaton()

        logger.info("[BaseGuard] Khởi tạo xong")

    def _has_blocked_keywords(self, text: str) -> bool:
        search_text = text.lower()
        for _ in self.blocker.iter(search_text):
            return True
        return False

    def _validate_lang_and_length(self, text: str) -> tuple[bool, str]:
        text_norm = unicodedata.normalize("NFC", text.strip())
        if not text_norm:
            return False, "Nội dung rỗng"

        labels, probs = self.language_model.predict(text_norm, k=1)
        lang = labels[0]
        conf = probs[0]

        if lang != self.allowed_lang or conf < self.lang_threshold:
            return False, "Nội dung vi phạm ngôn ngữ (chỉ hỗ trợ tiếng Việt)"

        if len(text) > self.max_chars:
            return False, "Nội dung vượt quá giới hạn ký tự"

        return True, ""

    async def _check_with_llm(self, prompt: str, user_input: str) -> str:
        try:
            response = await self.groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Bạn là trợ lý an toàn. TUYỆT ĐỐI CHỈ TRẢ LỜI 'Có' hoặc 'Không'. KHÔNG dùng <think>, KHÔNG giải thích, KHÔNG thêm text nào khác."},
                    {"role": "user", "content": prompt.format(relevant_text=user_input)}
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Lỗi gọi Groq: {e}")
            return "ERROR"

    async def check_input(self, user_input: str) -> tuple[bool, str]:
        # Bước 1 & 2: kiểm tra nhanh
        lang_length_ok, error_msg = self._validate_lang_and_length(user_input)
        if not lang_length_ok:
            return False, error_msg

        if self._has_blocked_keywords(user_input):
            return False, "Nội dung chứa các từ nhạy cảm hoặc bị cấm"

        # Bước 4: Kiểm tra LLM - dùng prompts từ file riêng
        checks = [
            (prompts.toxicity,          "Nội dung truy vấn có chứa yếu tố độc hại, xúc phạm, bạo lực, phân biệt đối xử, khiêu dâm, khuyến khích tự hại hoặc bất hợp pháp."),
            (prompts.topic_restriction, "Nội dung truy vấn không thuộc chủ đề về y tế, sức khỏe."),
            (prompts.prompt_injection,  "Nội dung có dấu hiệu thao túng hoặc jailbreak. Không được phép."),
            (prompts.code_injection,    "Nội dung chứa mã nguy hiểm hoặc injection. Không được phép."),
        ]

        for prompt_template, error_message in checks:
            result = await self._check_with_llm(prompt_template, user_input)
            if "Có" in result:
                return False, error_message

        # Pass hết
        return True, user_input