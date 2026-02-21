import chromadb
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from pymongo import MongoClient
import datetime
import os
from src.config.settings import settings

print("=== Build BM25 & lưu nodes vào MongoDocumentStore (MongoDB) - Tối ưu tiếng Việt ===")

# ────────────────────────────────────────────────
# 1. Kết nối Chroma và đọc toàn bộ nodes
# ────────────────────────────────────────────────
chroma_path = settings.paths.vector_db_dir
collection_name = settings.vector_store.collection_name
client_chroma = chromadb.PersistentClient(path=chroma_path)
collection = client_chroma.get_collection(collection_name)
print(f"Tổng vectors: {collection.count():,}")

BATCH_SIZE = 10000
total = collection.count()
all_nodes = []
offset = 0

while offset < total:
    print(f"Lấy batch {offset:,}...")
    results = collection.get(
        include=["documents", "metadatas"],
        limit=BATCH_SIZE,
        offset=offset
    )
    for doc_text, meta in zip(results["documents"], results["metadatas"]):        
        node = TextNode(text=doc_text or "", metadata=meta or {}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[])
        all_nodes.append(node)
    offset += BATCH_SIZE

print(f"Đã tạo {len(all_nodes):,} nodes")

# ────────────────────────────────────────────────
# 2. Kết nối MongoDB cho DocumentStore
# ────────────────────────────────────────────────
mongo_uri = settings.doc_store.uri 
db_name = settings.doc_store.db_name
namespace = settings.doc_store.namespace 

docstore = MongoDocumentStore.from_uri(
    uri=mongo_uri,
    db_name=db_name,
    namespace=namespace,
)

print(f"Đã kết nối MongoDocumentStore (namespace: {namespace})")

# ────────────────────────────────────────────────
# 3. Lưu toàn bộ nodes vào MongoDB
# ────────────────────────────────────────────────
print("Đang lưu nodes vào MongoDB (có thể mất vài phút với 341k nodes)...")
docstore.add_documents(all_nodes)
print(f"Đã lưu thành công {len(docstore.docs)} nodes vào MongoDB!")

# Optional: Kiểm tra mẫu
if docstore.docs:
    sample_node_id = list(docstore.docs.keys())[0]
    print(f"Ví dụ node id: {sample_node_id}")
    print(f"Text mẫu (50 ký tự): {docstore.get_node(sample_node_id).text[:50]}...")

# ────────────────────────────────────────────────
# 4. Build BM25Retriever từ docstore - TỐI ƯU TIẾNG VIỆT
# ────────────────────────────────────────────────
print("Đang build BM25 index từ docstore (có thể mất vài phút)...")

bm25_retriever = BM25Retriever.from_defaults(
    docstore=docstore,
    similarity_top_k=20,
    
    # Tối ưu tiếng Việt: KHÔNG stem, không dùng stopwords tiếng Anh
    stemmer=None,                  # Bỏ stemming hoàn toàn
    skip_stemming=True,            # Nếu version llama-index hỗ trợ (an toàn)
    language=None,                 # Không remove stopwords theo ngôn ngữ
    
    # Optional: Nếu version bm25s mới (≥0.2.6), có thể thêm (nếu lỗi thì bỏ)
    # non_ascii=True,              # Hỗ trợ UTF-8 tốt hơn (nếu có)
    # encoding="utf-8",
    
    # Tokenizer mặc định (?u)\b\w\w+\b đã ổn cho tiếng Việt có dấu
    # token_pattern=r"(?u)\b\w\w+\b",  # Giữ mặc định hoặc tùy chỉnh nếu cần
)

# Lưu BM25 ra disk để load nhanh (không rebuild mỗi lần)
BM25_PERSIST_DIR = settings.paths.bm25_persist_dir 
os.makedirs(BM25_PERSIST_DIR, exist_ok=True)
bm25_retriever.persist(path=BM25_PERSIST_DIR)
print(f"Đã lưu BM25 retriever (tiếng Việt optimized) vào: {BM25_PERSIST_DIR}")

print("\nHoàn tất! Load lại trong HybridRetriever như sau:")
