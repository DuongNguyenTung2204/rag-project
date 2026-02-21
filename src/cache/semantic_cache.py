import hashlib
import json
import redis
import numpy as np
from typing import Optional, Tuple
from llama_index.core.schema import TextNode
from langfuse import observe
import logging
logger = logging.getLogger(__name__)

class RedisSemanticCache:
    """
    Semantic Cache cho RAG y tế:
    - Cache response dựa trên cosine similarity của query.
    - Dùng Redis làm backend.
    - Embedding bằng bge-m3.
    """
    def __init__(
        self,
        redis_url: Optional[str] = None,
        embed_model=None,  # Truyền từ ngoài (từ HybridRetriever)
        similarity_threshold: float = 0.95,
        cache_ttl_days: int = 90,
    ):
        if embed_model is None:
            raise ValueError("embed_model (HuggingFaceEmbedding) phải được truyền vào khi khởi tạo RedisSemanticCache")
        
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.embed_model = embed_model
        self.threshold = similarity_threshold
        self.ttl_seconds = 3600 * 24 * cache_ttl_days

        logger.info(f"[RedisSemanticCache] Khởi tạo: threshold={self.threshold}, TTL={cache_ttl_days} ngày")

    def _get_embedding(self, text: str) -> np.ndarray:
        node = TextNode(text=text)
        embedding = self.embed_model.get_text_embedding(node.get_content())
        return np.array(embedding, dtype=np.float32)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        if vec1.size == 0 or vec2.size == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _get_cache_key(self, question: str) -> str:
        return f"rag:semantic:{hashlib.md5(question.encode()).hexdigest()}"

    @observe(name="semantic_cache_get")
    def get(self, question: str) -> Optional[Tuple[str, str]]:
        """
        Lấy từ cache nếu có query tương tự (cosine >= threshold).
        Trả về (response, original_question) nếu hit, None nếu miss.
        """
        question_embedding = self._get_embedding(question)

        # Lấy tất cả key cache
        keys = self.redis_client.keys("rag:semantic:*")
        best_score = 0.0
        best_response = None
        best_original_q = None

        for key in keys:
            cached_data = self.redis_client.get(key)
            if cached_data:
                data = json.loads(cached_data)
                cached_embedding = np.array(data["embedding"], dtype=np.float32)
                similarity = self._cosine_similarity(question_embedding, cached_embedding)

                if similarity >= self.threshold and similarity > best_score:
                    best_score = similarity
                    best_response = data["response"]
                    best_original_q = data["question"]

        if best_response:
            logger.debug(f"[RedisSemanticCache HIT] Similarity={best_score:.4f} với query gốc: {best_original_q}")
            return best_response, best_original_q

        logger.debug(f"[RedisSemanticCache MISS] Similarity={best_score:.4f} với query gốc: {best_original_q}")
        return None

    @observe(name="semantic_cache_set")
    def set(self, question: str, response: str):
        """Lưu query + response + embedding vào cache."""
        embedding = self._get_embedding(question).tolist()
        data = {
            "question": question,
            "response": response,
            "embedding": embedding
        }
        cache_key = self._get_cache_key(question)
        self.redis_client.setex(cache_key, self.ttl_seconds, json.dumps(data))
        logger.info(f"[RedisSemanticCache] Đã lưu cache cho query: {question[:50]}...")