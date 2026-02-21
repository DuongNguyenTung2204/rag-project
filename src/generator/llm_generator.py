import os
from typing import Optional
from groq import AsyncGroq
from dotenv import load_dotenv
from src.generator.prompt import PromptTemplate
import logging
logger = logging.getLogger(__name__)
load_dotenv()

class LLMGenerator:
    def __init__(
        self,
        model: str = "qwen/qwen3-32b",  
        temperature: float = 0.4,
        max_completion_tokens: int = 4000,
    ):
        """
        Khởi tạo client Groq với model Qwen.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY không được tìm thấy trong .env")

        self.client = AsyncGroq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        logger.info(f"[LLMGenerator] Khởi tạo Groq với model: {self.model}, Temperature: {self.temperature}, Max Tokens: {self.max_completion_tokens}")

    async def generate_response(
        self,
        question: str,
        retrieved_context: str,
    ) -> str:
        """
        Gọi Groq để sinh câu trả lời dựa trên prompt y tế an toàn.
        """
        # Sử dụng class PromptTemplate để xây messages
        messages = PromptTemplate.build_messages(
            question=question,
            retrieved_context=retrieved_context
        )

        logger.debug(f"Prompt sau khi thêm context: {messages}")
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
                reasoning_effort="none",
                top_p=0.9,
                stream=False,           # đổi thành True nếu muốn stream
                # Các tham số khác có thể thêm ở đây (xem phần giải thích bên dưới)
            )

            final_answer = response.choices[0].message.content.strip()
            logger.debug(f"[LLMGenerator] Đã sinh câu trả lời, độ dài: {len(final_answer)} ký tự")
            logger.debug(final_answer)
            return final_answer

        except Exception as e:
            logger.error(f"[LLMGenerator] Lỗi khi gọi Groq: {str(e)}")
            return (
                "Xin lỗi, hệ thống đang gặp sự cố khi xử lý câu hỏi. "
                "Vui lòng thử lại sau vài giây hoặc đặt câu hỏi khác nhé!"
            )


