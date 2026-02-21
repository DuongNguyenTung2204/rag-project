from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Literal, Optional


class PathsConfig(BaseSettings):
    """Các đường dẫn chung của project"""
    _this_file = Path(__file__).resolve()           # đường dẫn tuyệt đối tới file settings.py
    base_dir: Path = _this_file.parent.parent
    vector_db_dir: Path = base_dir / "storage" / "chroma_db"
    bm25_persist_dir: Path = base_dir/ "bm25_persist_vi"
    logs_dir: Path = base_dir / "logs"
    fasttext_model_dir: Path = base_dir / "models" / "lid.176.bin"
    blocked_file_path: Path = base_dir / "secrets" / "blocked_keywords.txt"
    embedded_nodes_path: Path = base_dir / "storage" / "embedded_nodes.pkl"
    dataset_dir: Path = base_dir.parent / "datasets"


class EmbeddingConfig(BaseSettings):
    """Cấu hình embedding model"""
    model_name: str = "BAAI/bge-m3"
    device: Literal["cuda", "cpu", "mps"] = "cuda"


class VectorStoreConfig(BaseSettings):
    """Cấu hình Chroma"""
    collection_name: str = "medical_healthcare_rag"


class DocStoreConfig(BaseSettings):
    """Cấu hình MongoDB Document Store"""
    uri: str = "mongodb://root:example123@localhost:27017"
    db_name: str = "rag_cache"
    namespace: str = "medical_rag_vi_2026"


class RetrieverConfig(BaseSettings):
    """Cấu hình retriever hybrid"""
    top_k_dense: int = 10
    top_k_bm25: int = 15
    top_k_final: int = 6
    use_rrf: bool = True


class LLMConfig(BaseSettings):
    """Cấu hình LLM (Groq)"""
    model: str = "qwen/qwen3-32b"
    small_model: str = "openai/gpt-oss-20b"
    guard_model: str = "openai/gpt-oss-safeguard-20b"

class ChainlitConfig(BaseSettings):
    """Cấu hình Chainlit / session history"""
    session_history_backend: Literal["memory", "redis", "file"] = "redis"
    redis_url: Optional[str] = "redis://localhost:6379/0"  # Default cho Docker trên Windows

class SemanticCacheConfig(BaseSettings):
    """Cấu hình Semantic Cache"""
    similarity_threshold: float = 0.9
    cache_ttl_days: int = 90

class AppConfig(BaseSettings):
    """Cấu hình chung ứng dụng"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    debug: bool = True
    log_level: str = "INFO"

    paths: PathsConfig = PathsConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    doc_store: DocStoreConfig = DocStoreConfig()
    retriever: RetrieverConfig = RetrieverConfig()
    llm: LLMConfig = LLMConfig()
    chainlit: ChainlitConfig = ChainlitConfig()
    semantic_cache: SemanticCacheConfig = SemanticCacheConfig()


# Instance global để import dễ dàng
settings = AppConfig()