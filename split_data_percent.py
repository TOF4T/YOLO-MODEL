import os
import shutil
import numpy as np
import argparse
from pathlib import Path
from collections import Counter

def get_image_labels(label_path):
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r', encoding='utf-8') as f:
        classes = [line.split()[0] for line in f.readlines() if line.strip()]
    return list(set(classes))

def allocate_split_custom(n):
    """
    Hàm phân phối số lượng ảnh:
    - Tính chuẩn tỷ lệ: 56% Train, 14% Valid, 30% Test trên TỔNG SỐ ảnh.
    - Ưu tiên dồn các ảnh dư do làm tròn vào tập Train.
    """
    if n <= 0: return 0, 0, 0
    if n == 1: return 1, 0, 0 
    if n == 2: return 1, 0, 1 
    if n == 3: return 1, 1, 1 

    # Tính theo tỷ lệ chuẩn trực tiếp trên n
    c_train = int(round(n * 0.56))
    c_valid = int(round(n * 0.14))
    c_test = int(round(n * 0.30))

    # Điều chỉnh nếu tổng bị lệch do sai số làm tròn
    current_total = c_train + c_valid + c_test
    
    while current_total > n:
        # Trừ bớt ở Test hoặc Valid trước để bảo vệ Train
        if c_test > 1 and c_test > int(n * 0.30): c_test -= 1
        elif c_valid > 1 and c_valid > int(n * 0.14): c_valid -= 1
        else: c_train -= 1
        current_total = c_train + c_valid + c_test
        
    while current_total < n:
        # Nếu thiếu ảnh so với n, CỘNG HẾT VÀO TRAIN
        c_train += 1
        current_total = c_train + c_valid + c_test
        
    return c_train, c_valid, c_test

def split_dataset(root_dir, output_dir):
    img_dir = os.path.join(root_dir, 'images')
    lbl_dir = os.path.join(root_dir, 'labels')
    
    if not os.path.exists(img_dir):
        print(f"❌ Không tìm thấy thư mục images tại: {img_dir}")
        return

    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG')
    all_files_with_ext = []
    for ext in valid_extensions:
        all_files_with_ext.extend(list(Path(img_dir).glob(f'*{ext}')))
    
    # 🟢 ĐÃ SỬA LỖI: Sắp xếp danh sách file chuẩn theo bảng chữ cái
    # Điều này đảm bảo tính tái lập tuyệt đối khi kết hợp với random.seed()
    all_files_with_ext.sort()
    
    all_files = [f.stem for f in all_files_with_ext]
    file_to_ext = {f.stem: f.suffix for f in all_files_with_ext}

    if not all_files:
        print(f"❌ Không tìm thấy file ảnh nào trong {img_dir}!")
        return

    all_classes_global = []
    file_to_classes = {}
    for f in all_files:
        lbl_path = os.path.join(lbl_dir, f + '.txt')
        classes = get_image_labels(lbl_path)
        file_to_classes[f] = classes
        all_classes_global.extend(classes)
        
    class_counts = Counter(all_classes_global)
    
    print("\n=== THỐNG KÊ SỐ LƯỢNG NHÃN BAN ĐẦU ===")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1]):
        label_id = int(cls) + 1
        print(f"Nhãn [{label_id}]: {count} ảnh")

    # Gom nhóm các file theo class hiếm nhất xuất hiện trong file đó
    grouped_files = {}
    for f in all_files:
        classes = file_to_classes[f]
        rarest_class = min(classes, key=lambda c: class_counts[c]) if classes else "background"
        if rarest_class not in grouped_files: 
            grouped_files[rarest_class] = []
        grouped_files[rarest_class].append(f)

    # Sắp xếp các nhóm: Ưu tiên các nhóm (class) có ít dữ liệu xử lý TRƯỚC
    sorted_groups = sorted(
        grouped_files.items(), 
        key=lambda x: class_counts[x[0]] if x[0] != "background" else float('inf')
    )

    train_files, valid_files, test_files = [], [], []
    
    # 🟢 GIỮ NGUYÊN: Cố định hạt giống sinh số ngẫu nhiên
    np.random.seed(42)
    
    print("\n=== TIẾN TRÌNH CHIA DATA THEO CLASS (ƯU TIÊN CLASS ÍT) ===")
    for group_name, files in sorted_groups:
        shuffled = files.copy()
        np.random.shuffle(shuffled)
        
        c_tr, c_va, c_te = allocate_split_custom(len(shuffled))
        
        train_files.extend(shuffled[:c_tr])
        valid_files.extend(shuffled[c_tr:c_tr+c_va])
        test_files.extend(shuffled[c_tr+c_va:])
        
        g_label = int(group_name) + 1 if group_name != "background" else "Background"
        print(f"Class [{g_label}] (Tổng {len(shuffled)}): Chia -> Train: {c_tr} | Valid: {c_va} | Test: {c_te}")

    for set_name, files in {'train': train_files, 'valid': valid_files, 'test': test_files}.items():
        img_out = os.path.join(output_dir, set_name, 'images')
        lbl_out = os.path.join(output_dir, set_name, 'labels')
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        
        for f in files:
            ext = file_to_ext[f]
            shutil.copy(os.path.join(img_dir, f + ext), os.path.join(img_out, f + ext))
            label_file = f + '.txt'
            if os.path.exists(os.path.join(lbl_dir, label_file)):
                shutil.copy(os.path.join(lbl_dir, label_file), os.path.join(lbl_out, label_file))

    # Tính toán tỷ lệ thực tế đạt được sau khi chia
    total = len(all_files)
    print("\n=== KẾT QUẢ CUỐI CÙNG ===")
    print(f"Thư mục lưu kết quả: {output_dir}")
    print(f"Tổng số ảnh xử lý: {total}")
    print(f" 🟢 Train: {len(train_files)} ảnh ({len(train_files)/total*100:.2f}%)")
    print(f" 🟡 Valid: {len(valid_files)} ảnh ({len(valid_files)/total*100:.2f}%)")
    print(f" 🔵 Test : {len(test_files)} ảnh ({len(test_files)/total*100:.2f}%)")
    print("✅ Hoàn tất! Tỷ lệ đã được bám sát chuẩn 56/14/30 và ưu tiên tối đa dữ liệu cho tập Train.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, required=True, help='Path to dataset (contains images/ and labels/)')
    parser.add_argument('--outpath', type=str, default='/content/Dataset_split', help='Path to save split data')
    args = parser.parse_args()
    
    split_dataset(args.datapath, args.outpath)
