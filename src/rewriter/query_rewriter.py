# rewriter/query_rewriter.py
from typing import List
from groq import AsyncGroq
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from dotenv import load_dotenv
from langfuse import observe
import logging
import os

logger = logging.getLogger(__name__)
load_dotenv()

class QueryRewriter:
    """
    History-aware query rewriting, gọi thẳng Groq client.
    Prompt đã được cải tiến với few-shot examples liên quan đến y tế.
    """

    def __init__(
        self,
        small_model: str,
    ):
        self.model = small_model
        self.groq_client = AsyncGroq(
            api_key=os.getenv("GROQ_API_KEY"),
        )
        
        self.system_prompt = """
Bạn là trợ lý chuyên viết lại câu hỏi tiếng Việt để phù hợp với tìm kiếm thông tin y tế.

NHIỆM VỤ CHÍNH:
- Biến câu hỏi hiện tại thành một câu hỏi ĐỘC LẬP, ĐẦY ĐỦ NGỮ CẢNH bằng tiếng Việt.
- Thay thế đại từ (này, đó, bệnh này, thuốc này, triệu chứng đó, cách chữa đó...) bằng thông tin cụ thể từ lịch sử hội thoại.
- Giữ nguyên ý nghĩa gốc, chỉ làm cho câu hỏi rõ ràng và tự chứa đựng đủ thông tin để trả lời mà không cần xem lịch sử.

QUY TẮC BẮT BUỘC:
- TOÀN BỘ output PHẢI bằng TIẾNG VIỆT, không chứa bất kỳ từ tiếng Anh nào.
- CHỈ TRẢ VỀ ĐÚNG MỘT CÂU HỎI đã viết lại. KHÔNG giải thích, KHÔNG thêm lời dẫn, KHÔNG trả lời nội dung câu hỏi.
- Nếu câu hỏi hiện tại đã hoàn toàn độc lập (không phụ thuộc lịch sử) → giữ nguyên nguyên vẹn.
- Nếu có đại từ hoặc từ ám chỉ → BẮT BUỘC thay thế bằng nội dung cụ thể từ lịch sử.

Ví dụ 1:
Lịch sử:
Người dùng: Triệu chứng của bệnh cảm cúm là gì?
Trợ lý: [trả lời về triệu chứng cảm cúm]
Người dùng: Cách chữa bệnh này tại nhà là gì?

→ Cách chữa bệnh cảm cúm tại nhà là gì?

Ví dụ 2:
Lịch sử:
Người dùng: Đau đầu do thiếu máu là gì?
Trợ lý: [giải thích về đau đầu thiếu máu]
Người dùng: Thuốc nào trị được triệu chứng này?

→ Thuốc nào trị được triệu chứng đau đầu do thiếu máu?

Ví dụ 3:
Lịch sử:
Người dùng: Viêm họng cấp có nên dùng kháng sinh không?
Trợ lý: [trả lời về viêm họng cấp]
Người dùng: Trường hợp nào thì cần dùng?

→ Trường hợp nào thì viêm họng cấp cần dùng kháng sinh?

Ví dụ 4:
Lịch sử:
Người dùng: Người bị tiểu đường type 2 nên ăn gì?
Trợ lý: [gợi ý chế độ ăn]
Người dùng: Loại trái cây nào tốt cho bệnh này?

→ Loại trái cây nào tốt cho người bị tiểu đường type 2?

Bắt đầu viết lại câu hỏi hiện tại dựa trên lịch sử trên:
""".strip()

    @observe(name="query_rewrite")
    async def rewrite(
        self,
        question: str,
        chat_history: List[BaseMessage]
    ) -> str:
        if not chat_history:
            logger.debug("Không có lịch sử → trả về câu hỏi gốc")
            return question

        logger.debug(f"Bắt đầu rewrite | lịch sử: {len(chat_history)} tin nhắn")

        messages: List[dict] = [{"role": "system", "content": self.system_prompt}]

        # Thêm lịch sử (chuyển đổi sang format OpenAI-compatible)
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "user"  # fallback an toàn

            # Giới hạn nội dung lịch sử nếu quá dài (tránh context overflow)
            content = msg.content[:800] if len(msg.content) > 800 else msg.content
            messages.append({"role": role, "content": content})

        # Câu hỏi cần rewrite
        messages.append({"role": "user", "content": question})

        try:
            response = await self.groq_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,           # nhẹ để linh hoạt hơn
                max_tokens=400,            # đủ chỗ cho câu dài
                top_p=0.95,
                stream=False,
            )

            rewritten = response.choices[0].message.content.strip()

            # Bảo vệ output: nếu không giống câu hỏi hoặc có dấu hiệu trả lời → fallback
            if (
                not rewritten
                or len(rewritten) < 5
                or "là trợ lý" in rewritten.lower()
                or "xin lỗi" in rewritten.lower()
                or rewritten.lower().startswith(("tôi", "bạn", "chào"))
            ):
                logger.warning("Output rewrite không hợp lệ, fallback về gốc")
                return question

            logger.debug(f"Kết quả rewrite: {rewritten}")
            return rewritten

        except Exception as e:
            logger.warning(f"[QueryRewriter] Lỗi khi gọi Groq: {e}", exc_info=True)
            return question  # fallback