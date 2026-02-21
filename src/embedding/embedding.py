from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.embeddings import BaseEmbedding
import logging

logger = logging.getLogger(__name__)

class EmbeddingProvider:
    """
    Class đơn giản để load và cung cấp embedding model.
    Chỉ load model khi khởi tạo instance.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
    ):
        logger.info(
            "[EmbeddingProvider] Bắt đầu load embedding model: %s trên device %s",
            model_name, device
        )

        try:
            self._embed_model = HuggingFaceEmbedding(
                model_name=model_name,
                device=device,
            )
            # Optional: kiểm tra dimension để debug
            test_dim = len(self._embed_model.get_text_embedding("kiểm tra"))
            logger.debug(
                "[EmbeddingProvider] Đã load thành công embedding model: %s | device: %s | dimension: %d",
                model_name, device, test_dim
            )
            logger.info("Embedding model sẵn sàng sử dụng.")

        except Exception as e:
            logger.error(
                "[EmbeddingProvider] Không thể load embedding model '%s' trên %s: %s",
                model_name, device, str(e),
                exc_info=True  # thêm stack trace để dễ debug
            )
            raise RuntimeError(f"Không thể load embedding model: {e}")

    @property
    def embed_model(self) -> BaseEmbedding:
        return self._embed_model