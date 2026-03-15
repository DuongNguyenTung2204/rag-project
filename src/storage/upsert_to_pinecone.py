import pickle
import os
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from tqdm import tqdm  # Để có progress bar đẹp (pip install tqdm nếu chưa có)
from src.config.settings import settings
from dotenv import load_dotenv
load_dotenv()
# ========================
# CẤU HÌNH
# ========================
API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = settings.pinecone.index_name
PKL_PATH = settings.paths.embedded_nodes_path

# Kiểm tra file tồn tại
if not os.path.exists(PKL_PATH):
    print(f"File không tồn tại: {PKL_PATH}")
    exit(1)

# ========================
# Bước 1: Khởi tạo Pinecone client
# ========================
pc = Pinecone(api_key=API_KEY)

# Kiểm tra index đã tồn tại chưa
existing_indexes = pc.list_indexes().names()  # Trả về list tên index
if INDEX_NAME in existing_indexes:
    print(f"Index '{INDEX_NAME}' đã tồn tại. Đang kết nối...")
else:
    print(f"Tạo index serverless mới: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,          # Xác nhận dimension của bge-m3 embedding trong .pkl
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"   # Starter/free tier chỉ hỗ trợ region này
        )
    )
    print("Index đã tạo xong. Chờ 30–60 giây để index sẵn sàng...")
    # Optional: import time; time.sleep(60)  # Nếu cần chờ lâu hơn

# Kết nối đến index
pinecone_index = pc.Index(INDEX_NAME)

# ========================
# Bước 2: Load nodes từ pickle
# ========================
print("Đang load nodes từ file pickle...")
with open(PKL_PATH, "rb") as f:
    nodes = pickle.load(f)

print(f"Tổng số nodes: {len(nodes):,}")
if len(nodes) == 0:
    print("File pickle rỗng! Kết thúc.")
    exit(1)

# ========================
# Bước 3: Upsert thủ công (chỉ chạy 1 lần, batch để ổn định)
# ========================
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

print("Đang upsert nodes vào Pinecone (batch 500 nodes)...")
batch_size = 500  # Có thể tăng lên 1000 nếu mạng tốt, nhưng 500 an toàn cho 173k nodes
total_nodes = len(nodes)

for start in tqdm(range(0, total_nodes, batch_size), desc="Upsert progress"):
    end = min(start + batch_size, total_nodes)
    batch_nodes = nodes[start:end]
    
    # add() sẽ dùng embedding sẵn có trong node.embedding, không gọi embed_model
    vector_store.add(batch_nodes)
    
    print(f"  → Đã upsert {end}/{total_nodes} nodes")

print("\nUpsert hoàn tất!")
print(f"Tổng vectors trong index: {pinecone_index.describe_index_stats()['total_vector_count']:,}")
print(f"Index name: {INDEX_NAME}")
print("→ Vào dashboard Pinecone (app.pinecone.io) → chọn index → xem 'Usage' để kiểm tra storage used (GB).")
print("Nếu ≤ 2 GB → free tier OK mãi mãi. Nếu vượt → delete index để tránh phí $50/tháng.")