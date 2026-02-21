from typing import Optional
from llama_index.storage.docstore.mongodb import MongoDocumentStore
import logging
logger = logging.getLogger(__name__)

class MongoDocumentStoreManager:
    """
    Quản lý kết nối và truy xuất DocumentStore từ MongoDB.
    Hỗ trợ lazy connect và kiểm tra số lượng nodes.
    """

    def __init__(
        self,
        uri: str,
        db_name: str,
        namespace: str,
    ):
        self.uri = uri
        self.db_name = db_name
        self.namespace = namespace
        self._docstore: Optional[MongoDocumentStore] = None

    def connect(self) -> MongoDocumentStore:
        """Kết nối đến MongoDocumentStore (lazy connect)."""
        if self._docstore is None:
            try:
                self._docstore = MongoDocumentStore.from_uri(
                    uri=self.uri,
                    db_name=self.db_name,
                    namespace=self.namespace,
                )
                doc_count = len(self._docstore.docs)
                logger.info(f"Đã kết nối Mongo DocumentStore - Số nodes: {doc_count:,}")
            except Exception as e:
                raise RuntimeError(f"Không thể kết nối Mongo DocumentStore: {e}")
        return self._docstore

    def get_docstore(self) -> MongoDocumentStore:
        """Trả về docstore đã kết nối."""
        return self.connect()