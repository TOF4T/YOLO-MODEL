import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from collections import Counter

# --- CẤU HÌNH ĐƯỜNG DẪN ---
root_dir = 'path/to/your/dataset' # Thư mục chứa images/ và labels/
output_dir = 'path/to/output_split'
train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1

def get_image_labels(label_path):
    """Đọc file label và trả về danh sách các class id duy nhất trong ảnh đó"""
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r') as f:
        # Mỗi dòng trong YOLO: <class_id> <x> <y> <w> <h>
        classes = [line.split()[0] for line in f.readlines()]
    return list(set(classes)) # Trả về unique classes trong 1 ảnh

def split_dataset():
    img_dir = os.path.join(root_dir, 'images')
    lbl_dir = os.path.join(root_dir, 'labels')
    
    # Lấy danh sách tên file (không bao gồm đuôi)
    all_files = [f.stem for f in Path(img_dir).glob('*.jpg')] # Thay .jpg bằng định dạng của bạn
    
    # Bước 1: Tạo "đại diện" cho mỗi ảnh dựa trên class để chia tầng
    # Vì 1 ảnh có nhiều class, ta lấy class xuất hiện ít nhất hoặc class chính làm đại diện
    file_labels = []
    for f in all_files:
        lbl_path = os.path.join(lbl_dir, f + '.txt')
        classes = get_image_labels(lbl_path)
        # Tạo chuỗi đại diện cho sự kết hợp class (ví dụ: "0_1" nếu ảnh có class 0 và 1)
        file_labels.append("_".join(sorted(classes)) if classes else "no_label")

    # Bước 2: Thực hiện chia tầng (Stratified Split)
    # Chia Train và phần còn lại (Val + Test)
    train_files, temp_files, _, temp_labels = train_test_split(
        all_files, file_labels, test_size=(1 - train_ratio), stratify=file_labels, random_state=42
    )
    
    # Chia phần còn lại thành Val và Test
    val_size_relative = val_ratio / (val_ratio + test_ratio)
    val_files, test_files = train_test_split(
        temp_files, test_size=(1 - val_size_relative), stratify=temp_labels, random_state=42
    )

    # Bước 3: Copy files vào thư mục đích
    sets = {'train': train_files, 'val': val_files, 'test': test_files}
    for set_name, files in sets.items():
        for f in files:
            # Tạo folder
            os.makedirs(os.path.join(output_dir, set_name, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, set_name, 'labels'), exist_ok=True)
            
            # Copy Image
            shutil.copy(os.path.join(img_dir, f + '.jpg'), os.path.join(output_dir, set_name, 'images'))
            # Copy Label
            shutil.copy(os.path.join(lbl_dir, f + '.txt'), os.path.join(output_dir, set_name, 'labels'))

    print(f"✅ Hoàn tất! Đã chia: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test.")

if __name__ == "__main__":
    split_dataset()
