from typing import Optional
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import BaseNode
from src.storage.document_store_mongo import MongoDocumentStoreManager
import logging
logger = logging.getLogger(__name__)

class BM25RetrieverBuilder:
    """
    Builder chuyên trách tạo BM25 Retriever với docstore từ MongoDB.
    """

    @staticmethod
    def build(
        persist_dir: str,
        mongo_uri: str,
        mongo_db_name: str,
        mongo_namespace: str,
        similarity_top_k: int = 15,
    ) -> BM25Retriever:
        """
        Load BM25 từ persist dir + gán docstore từ Mongo → trả về retriever sẵn sàng.
        """
        # Kết nối Mongo
        mongo_manager = MongoDocumentStoreManager(
            uri=mongo_uri,
            db_name=mongo_db_name,
            namespace=mongo_namespace,
        )
        docstore = mongo_manager.get_docstore()

        # Load BM25
        bm25_retriever = BM25Retriever.from_persist_dir(path=persist_dir)

        # Gán docstore (bắt buộc để lấy text node)
        bm25_retriever.docstore = docstore

        # Override top_k
        bm25_retriever.similarity_top_k = similarity_top_k

        logger.info(f"BM25 retriever sẵn sàng (top_k={similarity_top_k}, nodes={len(docstore.docs):,})")
        return bm25_retriever


class BM25RetrieverWrapper:
    """
    Wrapper tiện (dễ mở rộng sau này: custom scoring, filter...).
    """

    def __init__(self, retriever: BM25Retriever):
        self.retriever = retriever

    async def aretrieve(self, query: str, **kwargs):
        return await self.retriever.aretrieve(query, **kwargs)

    def retrieve(self, query: str, **kwargs):
        return self.retriever.retrieve(query, **kwargs)