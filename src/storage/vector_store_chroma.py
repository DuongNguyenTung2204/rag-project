import chromadb
from typing import Optional
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
import logging
logger = logging.getLogger(__name__)

class ChromaVectorStoreManager:
    """
    Quản lý kết nối, collection và index cho Chroma Vector Store.
    Hỗ trợ lazy loading và reuse.
    """

    def __init__(
        self,
        persist_path: str,
        collection_name: str,
        embed_model: Optional[BaseEmbedding] = None,
    ):
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.embed_model = embed_model

        self._client: Optional[chromadb.PersistentClient] = None
        self._vector_store: Optional[ChromaVectorStore] = None
        self._index: Optional[VectorStoreIndex] = None

    def connect(self) -> chromadb.PersistentClient:
        """Kết nối đến Chroma (lazy connect)."""
        if self._client is None:
            try:
                self._client = chromadb.PersistentClient(path=self.persist_path)
                logger.info(f"Đã kết nối Chroma tại: {self.persist_path}")
            except Exception as e:
                raise RuntimeError(f"Không thể kết nối Chroma: {e}")
        return self._client

    def get_collection(self):
        """Lấy collection (nếu không tồn tại thì raise lỗi, giống code gốc)."""
        client = self.connect()
        try:
            collection = client.get_collection(self.collection_name)
        except ValueError as e:
            raise ValueError(f"Collection '{self.collection_name}' không tồn tại trong Chroma: {e}")
        
        count = collection.count()
        logger.debug(f"Collection '{self.collection_name}': {count:,} vectors")
        return collection

    def build_vector_store(self) -> ChromaVectorStore:
        """Tạo ChromaVectorStore từ collection."""
        if self._vector_store is None:
            collection = self.get_collection()
            self._vector_store = ChromaVectorStore(chroma_collection=collection)
        return self._vector_store

    def build_index(self) -> VectorStoreIndex:
        """Tạo VectorStoreIndex từ vector store."""
        if self._index is None:
            if self.embed_model is None:
                raise ValueError("Cần cung cấp embed_model để build index")

            vector_store = self.build_vector_store()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                embed_model=self.embed_model,
            )
        return self._index

    def get_retriever(self, similarity_top_k: int = 10) -> VectorIndexRetriever:
        """Trả về VectorIndexRetriever đã cấu hình."""
        index = self.build_index()
        return VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
        )