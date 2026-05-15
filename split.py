import os
import shutil
import numpy as np
from pathlib import Path
from collections import Counter

# --- CẤU HÌNH ĐƯỜNG DẪN ---
root_dir = 'path/to/your/dataset'  # Thư mục chứa images/ và labels/
output_dir = 'path/to/output_split'

# Cấu hình tỷ lệ chia theo điều kiện
RATIO_MAJORITY = (0.7, 0.2, 0.1)  # Dành cho nhãn >= 100 (Train 70%, Val 20%, Test 10%)
RATIO_MINORITY = (0.4, 0.3, 0.3)  # Dành cho nhãn < 100  (Train 40%, Val 30%, Test 30%)

def get_image_labels(label_path):
    """Đọc file label và trả về danh sách các class id duy nhất trong ảnh đó"""
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r') as f:
        classes = [line.split()[0] for line in f.readlines()]
    return list(set(classes))

def allocate_split(n, r_train, r_val, r_test):
    """Tính toán số lượng file chính xác cho từng tập train/val/test, tránh lỗi làm tròn hụt file"""
    if n <= 0: return 0, 0, 0
    if n == 1: return 1, 0, 0
    if n == 2: return 1, 1, 0
    
    c_train = int(round(n * r_train))
    c_val = int(round(n * r_val))
    c_test = n - c_train - c_val
    
    # Đảm bảo không tập nào bị trống nếu tỷ lệ yêu cầu lớn hơn 0
    if c_train <= 0: c_train = 1
    if c_val <= 0 and r_val > 0: c_val = 1
    if c_test <= 0 and r_test > 0: c_test = 1
    
    # Điều chỉnh lại nếu tổng số lượng sau làm tròn bị lệch so với n
    current_total = c_train + c_val + c_test
    while current_total > n:
        if c_train > 1: c_train -= 1
        elif c_val > 1: c_val -= 1
        else: c_test -= 1
        current_total = c_train + c_val + c_test
    while current_total < n:
        c_train += 1
        current_total = c_train + c_val + c_test
        
    return c_train, c_val, c_test

def split_dataset():
    img_dir = os.path.join(root_dir, 'images')
    lbl_dir = os.path.join(root_dir, 'labels')
    
    # Quét toàn bộ danh sách ảnh (Hãy đổi đuôi .jpg thành định dạng ảnh của bạn nếu cần)
    all_files = [f.stem for f in Path(img_dir).glob('*.jpg')]
    
    # BƯỚC 1: Thống kê số lượng của từng class trên toàn bộ dataset
    all_classes_global = []
    file_to_classes = {}
    
    for f in all_files:
        lbl_path = os.path.join(lbl_dir, f + '.txt')
        classes = get_image_labels(lbl_path)
        file_to_classes[f] = classes
        all_classes_global.extend(classes)
        
    class_counts = Counter(all_classes_global)
    
    print("\n=== THỐNG KÊ SỐ LƯỢNG NHÃN BAN ĐẦU ===")
    for cls, count in sorted(class_counts.items()):
        # CẬP NHẬT: Đổi ngưỡng kiểm tra trực quan từ 50 thành 100
        status = "🟢 Số lượng Tốt (>=100) -> Chia 7:2:1" if count >= 100 else "🔴 Số lượng Ít (<100) -> Chia 4:3:3"
        print(f"Nhãn [{cls}]: {count} instance(s) | {status}")
    print("=======================================\n")

    # BƯỚC 2: Phân loại từng file ảnh dựa trên nhãn hiếm nhất có trong bức ảnh đó
    grouped_files = {}  # Cấu trúc: class_id -> [danh sách các file]
    
    for f in all_files:
        classes = file_to_classes[f]
        if not classes:
            rarest_class = "background"
        else:
            rarest_class = min(classes, key=lambda c: class_counts[c])
            
        if rarest_class not in grouped_files:
            grouped_files[rarest_class] = []
        grouped_files[rarest_class].append(f)

    # BƯỚC 3: Tiến hành chia theo tỷ lệ tương ứng cho từng nhóm nhãn
    train_files, val_files, test_files = [], [], []
    np.random.seed(42)
    
    for group_name, files in grouped_files.items():
        shuffled_files = files.copy()
        np.random.shuffle(shuffled_files)
        
        # Kiểm tra điều kiện số lượng nhãn của nhóm để quyết định tỷ lệ chia
        if group_name == "background":
            r_train, r_val, r_test = RATIO_MAJORITY
        else:
            # CẬP NHẬT: Đổi điều kiện lọc từ 50 thành 100 ở đây
            if class_counts[group_name] < 100:
                r_train, r_val, r_test = RATIO_MINORITY
            else:
                r_train, r_val, r_test = RATIO_MAJORITY
                
        # Tính toán phân phối số lượng
        n = len(shuffled_files)
        c_train, c_val, c_test = allocate_split(n, r_train, r_val, r_test)
        
        # Chia mảng và thêm vào danh sách tổng
        train_files.extend(shuffled_files[:c_train])
        val_files.extend(shuffled_files[c_train:c_train+c_val])
        test_files.extend(shuffled_files[c_train+c_val:])

    # BƯỚC 4: Tạo thư mục cấu trúc YOLO và copy file sang
    sets = {'train': train_files, 'val': val_files, 'test': test_files}
    for set_name, files in sets.items():
        for f in files:
            os.makedirs(os.path.join(output_dir, set_name, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, set_name, 'labels'), exist_ok=True)
            
            src_img = os.path.join(img_dir, f + '.jpg')
            dst_img = os.path.join(output_dir, set_name, 'images')
            src_lbl = os.path.join(lbl_dir, f + '.txt')
            dst_lbl = os.path.join(output_dir, set_name, 'labels')
            
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, dst_lbl)

    print("=== KẾT QUẢ PHÂN CHIA DATASET ===")
    print(f"Tổng số lượng ảnh đã chia: {len(all_files)}")
    print(f" └─ Tập Train: {len(train_files)} ảnh")
    print(f" └─ Tập Val:   {len(val_files)} ảnh")
    print(f" └─ Tập Test:  {len(test_files)} ảnh")
    print("✅ Hoàn tất! Dữ liệu đã chia theo ngưỡng điều kiện < 100.")

if __name__ == "__main__":
    split_dataset()
