import os
from typing import Optional, List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from src.embedding.embedding import EmbeddingProvider
from src.guards.input_guard import InputGuard
from src.rewriter.query_rewriter import QueryRewriter
from src.retrievers.hybrid import HybridRetriever
from src.generator.llm_generator import LLMGenerator
from src.cache.semantic_cache import RedisSemanticCache
from src.config.settings import settings
import logging
logger = logging.getLogger(__name__)
load_dotenv()

class Rag:
    def __init__(self):
        logger.info("[Rag] Đang khởi tạo pipeline...")
        
        self.input_guard = InputGuard(
            guard_model=settings.llm.guard_model,
            fasttext_model_dir=settings.paths.fasttext_model_dir,
            blocked_file_path=settings.paths.blocked_file_path,
        )

        self.query_rewriter = QueryRewriter(
            small_model=settings.llm.small_model,
        )

        self.embedding_provider = EmbeddingProvider(
            model_name=settings.embedding.model_name,
            device=settings.embedding.device,
        )           
        self.embed_model = self.embedding_provider.embed_model
        
        self.retriever = HybridRetriever(
            vector_db_dir=settings.paths.vector_db_dir,
            collection_name=settings.vector_store.collection_name,
            embed_model=self.embed_model,
            mongo_uri=settings.doc_store.uri,
            mongo_db_name=settings.doc_store.db_name,
            mongo_namespace=settings.doc_store.namespace,
            bm25_persist_dir=settings.paths.bm25_persist_dir,
            small_model=settings.llm.small_model,
            top_k_dense=settings.retriever.top_k_dense,
            top_k_bm25=settings.retriever.top_k_bm25,
            top_k_final=settings.retriever.top_k_final,
            use_rrf=settings.retriever.use_rrf,
        )

        self.generator = LLMGenerator(
            model=settings.llm.model,
        )

        self.semantic_cache = RedisSemanticCache(
            redis_url=settings.chainlit.redis_url,
            embed_model=self.embed_model,
            similarity_threshold=settings.semantic_cache.similarity_threshold,
            cache_ttl_days=settings.semantic_cache.cache_ttl_days,
        )

        logger.info("[Rag] Khởi tạo xong.")

    async def get_response(
        self,
        question: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        logger.debug(f"[Rag] Nhận câu hỏi: {question}")
        logger.debug(f"        session_id: {session_id}")
        logger.debug(f"        user_id: {user_id}")

        start_time = datetime.now()

        # Bước 1: Input Guardrail
        is_safe, processed_input_or_error = await self.input_guard.guard(question)
        if not is_safe:
            error_msg = (
                "Xin lỗi, câu hỏi của bạn không đáp ứng được các tiêu chuẩn an toàn hoặc phù hợp.\n"
                f"Lý do: {processed_input_or_error}\n\n"
                "Vui lòng thử lại với câu hỏi khác về sức khỏe hoặc y tế nhé!"
            )
            logger.debug(f"[Rag] Input bị từ chối: {processed_input_or_error}")
            return error_msg

        safe_question = processed_input_or_error
        logger.debug(f"[Rag] Input đã được anonymized và pass guardrail: {safe_question[:100]}...")

        # Bước 2: Rewrite query nếu có lịch sử
        rewritten_question = safe_question
      
        if len(chat_history) > 0:
            logger.info(f"[Rag] Có lịch sử chat: {len(chat_history)} tin nhắn")
            try:
                formatted_history = []
                for msg in chat_history:
                    if isinstance(msg, dict):
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if role == "user":
                            formatted_history.append(HumanMessage(content=content))
                        elif role == "assistant":
                            formatted_history.append(AIMessage(content=content))
                    else:
                        formatted_history.append(msg)

                rewritten_question = await self.query_rewriter.rewrite(
                    question=safe_question,
                    chat_history=formatted_history
                )
                logger.debug(f"[Rag] Query sau rewrite: {rewritten_question}")
            except Exception as e:
                logger.warning(f"[Rag] Lỗi khi rewrite query: {e}")
                rewritten_question = safe_question
        else:
            logger.debug("[Rag] Đây là tin nhắn đầu tiên → không rewrite, dùng query gốc")
        
        # Bước 2.1 Kiểm tra cache
        cached_result = self.semantic_cache.get(rewritten_question)
        if cached_result:
            response, original_q = cached_result
            logger.debug(f"[Semantic Cache HIT] Từ query gốc: {original_q}")
            time_taken = datetime.now() - start_time
            debug_info = f"\n\n(Thời gian xử lý: {time_taken.total_seconds():.2f}s | Cache HIT - không cần retrieve)"
            return response + debug_info

        # Bước 3: Retrieve
        try:
            nodes = await self.retriever.retrieve(rewritten_question)
            context = self.retriever.get_context_string(nodes)
            logger.info(f"[Rag] Retrieve thành công: {len(nodes)} nodes")
            logger.debug(context)
        except Exception as e:
            logger.warning(f"[Rag] Lỗi retrieve: {e}")
            context = "Không tìm thấy tài liệu liên quan."

        # Bước 4: Generate response bằng LLMGenerator
        try:
            final_response = await self.generator.generate_response(
                question=rewritten_question,
                retrieved_context=context
            )
            logger.info(f"[Rag] Generate thành công, độ dài: {len(final_response)} ký tự")
            # Lưu vào cache
            self.semantic_cache.set(rewritten_question, final_response)
            logger.info("[Semantic Cache] Đã lưu response mới")
        except Exception as e:
            logger.error(f"[Rag] Lỗi generate: {e}")
            final_response = (
                "Xin lỗi, hệ thống đang gặp sự cố khi sinh câu trả lời. "
                "Vui lòng thử lại sau vài giây hoặc đặt câu hỏi khác nhé!"
            )

        # Bước 5: Thời gian xử lý + debug info
        time_taken = datetime.now() - start_time
        debug_info = f"\n\n(Thời gian xử lý: {time_taken.total_seconds():.2f}s | Docs retrieved: {len(nodes) if 'nodes' in locals() else 0})"

        return final_response + debug_info


# Instance global
rag_service = Rag()