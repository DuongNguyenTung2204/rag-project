from typing import Optional
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.embeddings import BaseEmbedding
from src.storage.vector_store_pinecone import PineconeVectorStoreManager   # ← import class mới
import logging
logger = logging.getLogger(__name__)

class DenseRetrieverBuilder:
    """
    Builder chuyên trách tạo Dense Retriever (vector-based) từ Pinecone.
    """

    @staticmethod
    def build(
        api_key: str,
        index_name: str,
        embed_model: BaseEmbedding,
        similarity_top_k: int = 10,
        namespace: Optional[str] = None,
        text_key: str = "text",
    ) -> BaseRetriever:
        """
        Tạo và trả về retriever từ Pinecone index đã tồn tại.
        """
        manager = PineconeVectorStoreManager(
            api_key=api_key,
            index_name=index_name,
            embed_model=embed_model,
            namespace=namespace,
            text_key=text_key,
        )

        retriever = manager.get_retriever(similarity_top_k=similarity_top_k)
        logger.info(f"Dense retriever (Pinecone) sẵn sàng (top_k={similarity_top_k})")
        return retriever


class DenseRetriever:
    """
    Wrapper tiện (nếu sau này cần thêm logic như post-processing, caching...).
    Hiện tại chỉ delegate cho retriever gốc.
    """

    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever

    async def aretrieve(self, query: str, **kwargs):
        return await self.retriever.aretrieve(query, **kwargs)

    def retrieve(self, query: str, **kwargs):
        return self.retriever.retrieve(query, **kwargs)