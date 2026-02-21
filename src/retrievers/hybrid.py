import os
from typing import List, Optional
from llama_index.core.retrievers import QueryFusionRetriever, BaseRetriever
from llama_index.core.schema import NodeWithScore

from src.retrievers.dense import DenseRetrieverBuilder
from src.retrievers.bm25 import BM25RetrieverBuilder

from llama_index.core import Settings
from llama_index.llms.groq import Groq
from langfuse import observe

import logging
logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Hybrid Retriever kết hợp Dense + BM25 bằng QueryFusionRetriever (RRF).
    """

    def __init__(
        self,
        # Các tham số config (có thể truyền từ config object)
        vector_db_dir: str,
        collection_name: str,
        embed_model,
        mongo_uri: str,
        mongo_db_name: str,
        mongo_namespace: str,
        bm25_persist_dir: str,
        small_model: str,
        top_k_dense: int = 10,
        top_k_bm25: int = 15,
        top_k_final: int = 6,
        use_rrf: bool = True,    
    ):
        # Set LLM Groq (từ .env)
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("Không tìm thấy GROQ_API_KEY trong .env")

        Settings.llm = Groq(
            model=small_model,  # hoặc model Groq khác nếu bạn muốn
            api_key=groq_api_key,
            temperature=0.1,
        )
        logger.info("Đã set LLM Groq cho QueryFusionRetriever")

        # Tạo dense retriever
        self.dense_retriever = DenseRetrieverBuilder.build(
            persist_path=vector_db_dir,
            collection_name=collection_name,
            embed_model=embed_model,
            similarity_top_k=top_k_dense,
        )

        # Tạo bm25 retriever
        self.bm25_retriever = BM25RetrieverBuilder.build(
            persist_dir=bm25_persist_dir,
            mongo_uri=mongo_uri,
            mongo_db_name=mongo_db_name,
            mongo_namespace=mongo_namespace,
            similarity_top_k=top_k_bm25,
        )

        # Fusion
        logger.info("Khởi tạo QueryFusionRetriever (hybrid mode)...")
        self.fusion_retriever = QueryFusionRetriever(
            retrievers=[self.dense_retriever, self.bm25_retriever],
            similarity_top_k=top_k_final,
            num_queries=1,
            mode="reciprocal_rerank" if use_rrf else "simple",
            use_async=True,
            verbose=True,
        )

    @observe(name="hybrid_retrieve")
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[NodeWithScore]:
        if top_k is not None:
            original_k = self.fusion_retriever.similarity_top_k
            self.fusion_retriever.similarity_top_k = top_k
            nodes = await self.fusion_retriever.aretrieve(query)
            self.fusion_retriever.similarity_top_k = original_k
        else:
            nodes = await self.fusion_retriever.aretrieve(query)
        return nodes

    def get_context_string(self, nodes: List[NodeWithScore], max_chars: int = 15000) -> str:
        if not nodes:
            return "Không tìm thấy tài liệu liên quan."

        sorted_nodes = sorted(nodes, key=lambda x: x.score if x.score is not None else float('-inf'), reverse=True)

        context_parts = []
        current_length = 0

        header = "Các nguồn tham khảo (sắp xếp theo độ liên quan cao nhất):\n\n"
        context_parts.append(header)
        current_length += len(header)

        for i, node_score in enumerate(sorted_nodes, 1):
            node = node_score.node
            score = node_score.score if node_score.score is not None else "N/A"

            text = node.get_content(metadata_mode="none").strip()
            title = node.metadata.get("title", "Không có tiêu đề")
            url = node.metadata.get("url", node.metadata.get("source_url", "Không có link"))

            # Format rõ ràng để LLM dễ map [i]
            source_info = f"[{i}] {title} - {url} (Score: {score:.4f})"
            segment = f"{source_info}\n{text}\n{'-'*60}\n"

            block_length = len(segment)
            if current_length + block_length > max_chars:
                break

            context_parts.append(segment)
            current_length += block_length

        return "".join(context_parts)
