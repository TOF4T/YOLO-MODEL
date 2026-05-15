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

def allocate_split(n, r_train, r_val, r_test):
    if n <= 0: return 0, 0, 0
    if n == 1: return 1, 0, 0
    c_train = int(round(n * r_train))
    c_val = int(round(n * r_val))
    c_test = n - c_train - c_val
    if c_train <= 0: c_train = 1
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

def split_dataset(root_dir, output_dir):
    img_dir = os.path.join(root_dir, 'images')
    lbl_dir = os.path.join(root_dir, 'labels')
    
    # Kiểm tra thư mục đầu vào
    if not os.path.exists(img_dir):
        print(f"❌ Không tìm thấy thư mục images tại: {img_dir}")
        return

    # Tự động tìm nhiều định dạng ảnh phổ biến (jpg, png, jpeg, JPG...)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG')
    all_files_with_ext = []
    for ext in valid_extensions:
        all_files_with_ext.extend(list(Path(img_dir).glob(f'*{ext}')))
    
    all_files = [f.stem for f in all_files_with_ext]
    file_to_ext = {f.stem: f.suffix for f in all_files_with_ext}

    if not all_files:
        print(f"❌ Không tìm thấy file ảnh nào trong {img_dir}. Hãy kiểm tra định dạng ảnh!")
        return

    # Tỷ lệ chia
    RATIO_MAJORITY = (0.7, 0.2, 0.1)
    RATIO_MINORITY = (0.4, 0.3, 0.3)

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
        status = "🟢 >=100 (7:2:1)" if count >= 100 else "🔴 <100 (4:3:3)"
        print(f"Nhãn [{cls}]: {count} | {status}")

    grouped_files = {}
    for f in all_files:
        classes = file_to_classes[f]
        rarest_class = min(classes, key=lambda c: class_counts[c]) if classes else "background"
        if rarest_class not in grouped_files: grouped_files[rarest_class] = []
        grouped_files[rarest_class].append(f)

    train_files, val_files, test_files = [], [], []
    np.random.seed(42)
    
    for group_name, files in grouped_files.items():
        shuffled = files.copy()
        np.random.shuffle(shuffled)
        r = RATIO_MINORITY if (group_name != "background" and class_counts[group_name] < 100) else RATIO_MAJORITY
        c_tr, c_va, c_te = allocate_split(len(shuffled), *r)
        train_files.extend(shuffled[:c_tr])
        val_files.extend(shuffled[c_tr:c_tr+c_va])
        test_files.extend(shuffled[c_tr+c_va:])

    # Copy files
    for set_name, files in {'train': train_files, 'val': val_files, 'test': test_files}.items():
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

    print(f"\n=== KẾT QUẢ: Chia {len(all_files)} ảnh thành {len(train_files)} Train, {len(val_files)} Val, {len(test_files)} Test. ✅")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, required=True, help='Path to dataset (contains images/ and labels/)')
    parser.add_argument('--outpath', type=str, default='/content/dataset_split', help='Path to save split data')
    args = parser.parse_args()
    
    split_dataset(args.datapath, args.outpath)
