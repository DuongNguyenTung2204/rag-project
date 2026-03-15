import os
from typing import Optional, Any
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
import logging

logger = logging.getLogger(__name__)

class PineconeVectorStoreManager:
    """
    Quản lý kết nối và truy vấn Pinecone index đã tồn tại.
    - Lazy loading client, index, vector_store, và llama_index.
    - Tích hợp kiểm tra dimension để tránh lỗi mismatch.
    - Dùng cho dense retrieval trong RAG (tương thích HybridRetriever).
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        embed_model: Optional[BaseEmbedding] = None,
        namespace: Optional[str] = None,
        text_key: str = "text",  # Thay đổi nếu field text trong metadata là khác (ví dụ: "_node_content")
    ):
        self.api_key = api_key
        self.index_name = index_name
        self.embed_model = embed_model
        self.namespace = namespace
        self.text_key = text_key

        self._client: Optional[Pinecone] = None
        self._pinecone_index: Optional[Any] = None  # type: ignore
        self._vector_store: Optional[PineconeVectorStore] = None
        self._index: Optional[VectorStoreIndex] = None

    def connect(self) -> Pinecone:
        if self._client is None:
            try:
                self._client = Pinecone(api_key=self.api_key)
                logger.info("Đã kết nối Pinecone client thành công")
            except Exception as e:
                raise RuntimeError(f"Kết nối Pinecone thất bại: {e}")
        return self._client

    def get_pinecone_index(self) -> Any:  # type: ignore
        if self._pinecone_index is None:
            client = self.connect()
            try:
                self._pinecone_index = client.Index(self.index_name)
                stats = self._pinecone_index.describe_index_stats()
                total = stats.get('total_vector_count', 0)
                namespaces = list(stats.get('namespaces', {}).keys())
                logger.info(
                    f"Index '{self.index_name}' tồn tại | "
                    f"Tổng vectors: {total:,} | Namespaces: {namespaces}"
                )
            except Exception as e:
                raise ValueError(f"Index '{self.index_name}' không tồn tại hoặc lỗi: {e}")
        return self._pinecone_index

    def _check_dimension_match(self):
        """Kiểm tra dimension của embed_model có khớp với index không."""
        if self.embed_model is None:
            logger.warning("Chưa có embed_model → bỏ qua kiểm tra dimension")
            return

        # Lấy dimension từ embed_model (thử embed 1 chuỗi ngắn)
        test_embedding = self.embed_model.get_text_embedding("test")
        model_dim = len(test_embedding)

        # Lấy dimension từ Pinecone index
        index = self.get_pinecone_index()
        stats = index.describe_index_stats()
        index_dim = stats.get('dimension')

        if index_dim is None:
            logger.warning("Không lấy được dimension từ index stats")
            return

        if model_dim != index_dim:
            raise ValueError(
                f"Dimension mismatch! Embed model: {model_dim}d | "
                f"Pinecone index '{self.index_name}': {index_dim}d\n"
                "→ Hãy đảm bảo embed_model khớp với index (ví dụ: bge-m3 = 1024)"
            )
        logger.info(f"Dimension khớp: {model_dim}d")

    def build_vector_store(self) -> PineconeVectorStore:
        if self._vector_store is None:
            pinecone_index = self.get_pinecone_index()
            self._vector_store = PineconeVectorStore(
                pinecone_index=pinecone_index,
                namespace=self.namespace,
                text_key=self.text_key,
            )
            logger.debug(f"Đã tạo PineconeVectorStore (text_key='{self.text_key}', namespace={self.namespace})")
        return self._vector_store

    def build_index(self) -> VectorStoreIndex:
        if self._index is None:
            if self.embed_model is None:
                raise ValueError("Cần cung cấp embed_model để build VectorStoreIndex")

            # Kiểm tra dimension trước khi build
            self._check_dimension_match()

            vector_store = self.build_vector_store()
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self.embed_model,
            )
            logger.info("Đã load VectorStoreIndex từ Pinecone thành công")
        return self._index

    def get_retriever(self, similarity_top_k: int = 10) -> BaseRetriever:
        index = self.build_index()
        retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        logger.info(f"Dense retriever (Pinecone) sẵn sàng | top_k={similarity_top_k}")
        return retriever

    # Tiện ích debug (tương tự code test của bạn)
    def print_index_stats(self):
        index = self.get_pinecone_index()
        stats = index.describe_index_stats()
        print(f"Index '{self.index_name}' stats:")
        print(f"  Tổng vectors: {stats.get('total_vector_count', 0):,}")
        print(f"  Dimension: {stats.get('dimension', 'N/A')}")
        print(f"  Namespaces: {list(stats.get('namespaces', {}).keys())}")