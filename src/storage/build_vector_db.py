import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import time
import torch
from tqdm import tqdm

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.schema import TextNode
from src.config.settings import settings
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from transformers import AutoTokenizer
import pickle

# ========================
# CẤU HÌNH
# ========================
INPUT_FOLDER = settings.paths.dataset_dir
PERSIST_DIR = settings.paths.vector_db_dir 
collection_name = settings.vector_store.collection_name
EMBEDDED_NODES_PATH = settings.paths.embedded_nodes_path # Lưu để resume nếu crash
os.makedirs(PERSIST_DIR, exist_ok=True)

# Model & tokenizer
EMBED_MODEL_NAME = settings.embedding.model_name 
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME, device="cuda")
Settings.embed_model = embed_model

# Ngưỡng token
MAX_TOKEN_SINGLE_CHUNK = 1000

# Batch size embedding (tối ưu cho T4 16GB)
BATCH_SIZE_EMBED = 256  # Có thể tăng lên 64 nếu không OOM

# ========================
# HÀM HỖ TRỢ
# ========================
def count_tokens(text: str) -> int:
    if pd.isna(text) or not isinstance(text, str):
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))

def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed batch để tăng tốc"""
    if not texts:
        return []
    embeddings = embed_model.get_text_embedding_batch(texts, show_progress=False)
    torch.cuda.empty_cache()  # Giải phóng VRAM
    return embeddings

# ========================
# 1. Load documents (như cũ)
# ========================
documents = []

# medical_qa.csv - Ghép title + content
medical_qa_path = os.path.join(INPUT_FOLDER, "medical_qa.csv")
df_mqa = pd.read_csv(medical_qa_path)
for _, row in df_mqa.iterrows():
    question = row['title'] if pd.notna(row['title']) else ""
    answer = row['content'] if pd.notna(row['content']) else ""
    combined = f"Câu hỏi: {question.strip()}\nTrả lời: {answer.strip()}"
    metadata = {
        "source": "medical_qa",
        "question": question.strip(),
        "url": row.get('url', '')
    }
    documents.append(Document(text=combined, metadata=metadata))

# vinmec_article_main.csv - Chỉ content
main_path = os.path.join(INPUT_FOLDER, "vinmec_article_main.csv")
df_main = pd.read_csv(main_path)
for _, row in df_main.iterrows():
    text = row['content'] if pd.notna(row['content']) else ""
    metadata = {
        "source": "vinmec_article_main",
        "title": row.get('title', ''),
        "url": row.get('url', '')
    }
    documents.append(Document(text=text, metadata=metadata))

# vinmec_article_subtitle.csv - Chỉ content
subtitle_path = os.path.join(INPUT_FOLDER, "vinmec_article_subtitle.csv")
df_sub = pd.read_csv(subtitle_path)
for _, row in df_sub.iterrows():
    text = row['content'] if pd.notna(row['content']) else ""
    metadata = {
        "source": "vinmec_article_subtitle",
        "title": row.get('title', ''),
        "url": row.get('url', '')
    }
    documents.append(Document(text=text, metadata=metadata))

print(f"Tổng số documents: {len(documents):,}")

# ========================
# 2. Hybrid Chunking + Batch Embedding (xử lý dài trước, ngắn sau + tqdm)
# ========================
basic_splitter = SentenceSplitter(chunk_size=MAX_TOKEN_SINGLE_CHUNK, chunk_overlap=200)
semantic_splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=90,
    embed_model=embed_model  # Chỉ dùng để split
)

all_nodes = []

short_texts = []
short_metadatas = []
long_docs = []

# Phân loại short/long với tqdm
for doc in tqdm(documents, desc="Phân loại short/long"):
    token_count = count_tokens(doc.text)
    if token_count <= MAX_TOKEN_SINGLE_CHUNK:
        short_texts.append(doc.text)
        short_metadatas.append(doc.metadata)
    else:
        long_docs.append(doc)

print(f"Số mẫu ngắn (<=1000 token): {len(short_texts):,}")
print(f"Số mẫu dài (>1000 token): {len(long_docs):,}")

# XỬ LÝ MẪU DÀI TRƯỚC
print("\nXử lý mẫu dài (split semantic + embed batch)...")
long_nodes = []
for doc in tqdm(long_docs, desc="Split & embed mẫu dài"):
    sub_docs = semantic_splitter.get_nodes_from_documents([doc])  # Chỉ split
    sub_texts = [sub.text for sub in sub_docs]
    
    # Embed batch cho sub-nodes
    sub_embeddings = embed_batch(sub_texts)
    
    for sub_text, sub_emb in zip(sub_texts, sub_embeddings):
        node = TextNode(text=sub_text, metadata=doc.metadata)
        node.embedding = sub_emb
        long_nodes.append(node)

all_nodes.extend(long_nodes)

# XỬ LÝ MẪU NGẮN SAU
print("\nXử lý mẫu ngắn (embed batch)...")
short_embeddings = embed_batch(short_texts)
for text, emb, meta in tqdm(zip(short_texts, short_embeddings, short_metadatas), desc="Tạo node mẫu ngắn", total=len(short_texts)):
    node = TextNode(text=text, metadata=meta)
    node.embedding = emb
    all_nodes.append(node)

print(f"Tổng số nodes sau chunking + embedding: {len(all_nodes):,}")

# Lưu nodes đã embed (để resume nếu cần)
with open(EMBEDDED_NODES_PATH, "wb") as f:
    pickle.dump(all_nodes, f)
print(f"Đã lưu nodes đã embed vào {EMBEDDED_NODES_PATH}")

# ========================
# 3. Lưu vào Chroma
# ========================
db = chromadb.PersistentClient(path=PERSIST_DIR)
chroma_collection = db.get_or_create_collection(collection_name)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes=all_nodes,
    storage_context=storage_context,
    embed_model=None  # Không embed lại
)

print("Embedding và lưu vector DB xong!")
print(f"Vector DB lưu tại: {PERSIST_DIR}")
print(f"Số vector trong collection: {chroma_collection.count()}")