import os
from datasets import load_dataset
import pandas as pd
from src.config.settings import settings
# ========================
# CẤU HÌNH 
# ========================
output_folder = settings.paths.dataset_dir  
# Nếu folder chưa tồn tại, sẽ tự tạo
os.makedirs(output_folder, exist_ok=True)

# Danh sách 5 splits cần tải
splits = [
    "vinmec_article_subtitle",
    "medical_qa",
    "full",
    "vinmec_article_content",
    "vinmec_article_main"
]

# ========================
# TẢI VÀ LƯU TỪNG SPLIT
# ========================
print("Đang tải dataset urnus11/Vietnamese-Healthcare...")

for split_name in splits:
    print(f"\n→ Đang tải split: {split_name}")
    
    # Tải chỉ split đó (không tải toàn bộ dataset để tiết kiệm)
    dataset_split = load_dataset("urnus11/Vietnamese-Healthcare", split=split_name)
    
    # Chuyển thành pandas DataFrame
    df = dataset_split.to_pandas()
    
    # Tên file CSV: ví dụ vinmec_article_subtitle.csv
    output_file = os.path.join(output_folder, f"{split_name}.csv")
    
    # Lưu thành CSV (utf-8 để hỗ trợ tiếng Việt)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"Đã lưu thành công: {output_file}")
    print(f"Số dòng: {len(df):,} rows")
    print(f"Cột có sẵn: {list(df.columns)}")

print("\nHoàn tất! Tất cả 5 file CSV đã được lưu vào folder:")
print(output_folder)