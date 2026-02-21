from typing import Optional
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.embeddings import BaseEmbedding
from src.storage.vector_store_chroma import ChromaVectorStoreManager
import logging
logger = logging.getLogger(__name__)

class DenseRetrieverBuilder:
    """
    Builder chuyên trách tạo Dense Retriever (vector-based) từ Chroma.
    """

    @staticmethod
    def build(
        persist_path: str,
        collection_name: str,
        embed_model: BaseEmbedding,
        similarity_top_k: int = 10,
    ) -> BaseRetriever:
        """
        Tạo và trả về VectorIndexRetriever đã sẵn sàng.
        - Load từ Chroma persist dir (không rebuild index).
        """
        manager = ChromaVectorStoreManager(
            persist_path=persist_path,
            collection_name=collection_name,
            embed_model=embed_model,
        )

        retriever = manager.get_retriever(similarity_top_k=similarity_top_k)
        logger.info(f"Dense retriever sẵn sàng (top_k={similarity_top_k})")
        return retriever


class DenseRetriever:
    """
    Wrapper tiện (nếu sau này cần thêm logic như post-processing, caching...).
    Hiện tại chỉ delegate cho VectorIndexRetriever.
    """

    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever

    async def aretrieve(self, query: str, **kwargs):
        return await self.retriever.aretrieve(query, **kwargs)

    def retrieve(self, query: str, **kwargs):
        return self.retriever.retrieve(query, **kwargs)