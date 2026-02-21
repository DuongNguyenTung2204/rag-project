# app.py (file Chainlit chính)
import chainlit as cl
import logging
from langfuse import Langfuse  
from dotenv import load_dotenv
import os

load_dotenv()

# Khởi tạo Langfuse ONE TIME duy nhất
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)
logger = logging.getLogger(__name__)

# Import config logging và gọi ngay đầu file
from src.config.logging_config import setup_logging
setup_logging()

from langchain_community.chat_message_histories import RedisChatMessageHistory, ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from datetime import datetime
from typing import List

# Import pipeline RAG
from src.pipeline.rag_pipeline import rag_service

from src.config.settings import settings

@cl.on_chat_start
async def on_chat_start():
    logger.info("Bắt đầu phiên chat mới")
    await cl.Message(
        content="Chào bạn! Đây là chatbot RAG y tế thông minh.\n"
                "Mình sẽ trả lời dựa trên tài liệu y khoa đáng tin cậy (Vinmec, WHO, v.v.).\n"
                "Hỏi mình bất cứ điều gì về sức khỏe nhé! 🚀\n\n"
                "Lưu ý: Đây chỉ là thông tin tham khảo. Hãy tham khảo ý kiến bác sĩ để được tư vấn chính xác."
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    logger.info(f"Nhận tin nhắn từ user: {message.content[:100]}...")

    session_id = cl.context.session.id

    # Chọn backend history dựa trên settings
    if settings.chainlit.session_history_backend == "redis":
        if not settings.chainlit.redis_url:
            logger.error("Redis được chọn nhưng redis_url chưa được cấu hình trong settings")
            await cl.Message(content="❌ Lỗi hệ thống: Redis chưa được cấu hình.").send()
            return

        history = RedisChatMessageHistory(
            session_id=session_id,
            url=settings.chainlit.redis_url,
            ttl=3600 * 24 * 7
        )
        logger.debug(f"Sử dụng Redis history với URL: {settings.chainlit.redis_url}")
    else:
        history = ChatMessageHistory()
        logger.debug("Sử dụng in-memory history")

    # Hiển thị lịch sử cũ
    history_text = "**Lịch sử chat đến trước tin nhắn này:**\n\n"
    if not history.messages:
        history_text += "*(Đây là tin nhắn đầu tiên trong phiên này)*\n"
    else:
        for msg in history.messages:
            ts = msg.additional_kwargs.get("timestamp", "—")
            role = "Bạn" if isinstance(msg, HumanMessage) else "Bot"
            content = msg.content.strip()
            history_text += f"**{role}** • {ts}\n{content}\n{'─' * 50}\n\n"

    await cl.Message(content=history_text).send()

    # Lưu tin nhắn người dùng
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_msg = HumanMessage(
        content=message.content,
        additional_kwargs={"timestamp": now}
    )
    history.add_message(user_msg)

    try:
        loading_msg = cl.Message(content="Đang tìm kiếm và suy nghĩ...")
        await loading_msg.send()

        # Gọi pipeline RAG
        response = await rag_service.get_response(
            question=message.content,
            session_id=session_id,
            chat_history=history.messages,
        )

        loading_msg.content = response
        loading_msg.author = "Bot"
        await loading_msg.update()

        logger.info(f"Trả lời thành công, độ dài response: {len(response)} ký tự")

        bot_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        bot_msg = AIMessage(
            content=response,
            additional_kwargs={"timestamp": bot_now}
        )
        history.add_message(bot_msg)

    except Exception as e:
        logger.error(f"Lỗi khi xử lý tin nhắn: {str(e)}", exc_info=True)
        loading_msg.content = f"❌ Xin lỗi, có lỗi xảy ra khi xử lý:\n{str(e)}\nVui lòng thử lại nhé!"
        loading_msg.author = "Bot"
        await loading_msg.update()