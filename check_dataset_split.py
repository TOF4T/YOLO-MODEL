import os
from pathlib import Path
from collections import Counter

def count_labels_in_splits(split_root):
    """
    Quét qua các thư mục train, val, test và đếm số lượng instance cũng như số ảnh của từng class
    """
    splits = ['train', 'valid', 'test']

    # Tạo 2 từ điển để lưu trữ: 1 cho instance (đối tượng), 1 cho số lượng ảnh
    instance_stats = {split: Counter() for split in splits}
    image_stats = {split: Counter() for split in splits}

    print(f"🔍 Đang kiểm tra dữ liệu tại: {split_root}")
    print("-" * 50)

    for split in splits:
        label_dir = os.path.join(split_root, split, 'labels')

        if not os.path.exists(label_dir):
            print(f"⚠️ Cảnh báo: Không tìm thấy thư mục nhãn cho tập {split} tại {label_dir}")
            continue

        label_files = list(Path(label_dir).glob('*.txt'))

        for lbl in label_files:
            # Dùng set để lưu các class xuất hiện trong bức ảnh này (tránh đếm trùng lớp trong cùng 1 ảnh)
            classes_in_image = set() 
            
            with open(lbl, 'r') as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        try:
                            # Đếm số lượng instances (bbox)
                            class_id = int(float(parts[0])) 
                            instance_stats[split][class_id] += 1
                            
                            # Ghi nhận class này có xuất hiện trong ảnh hiện tại
                            classes_in_image.add(class_id)
                        except ValueError:
                            continue # Bỏ qua nếu dòng bị lỗi hoặc không phải số

            # Cập nhật đếm số lượng ảnh cho mỗi class có mặt trong bức ảnh này
            for cls_id in classes_in_image:
                image_stats[split][cls_id] += 1

    all_classes = sorted(list(set().union(*(s.keys() for s in instance_stats.values()))))

    header = f"{'Class ID':<10} | {'Train':<10} | {'Valid':<10} | {'Test':<10} | {'Total':<10}"
    
    # ---------------------------------------------------------
    # BẢNG 1: IN BẢNG SỐ LƯỢNG INSTANCE
    # ---------------------------------------------------------
    print("\n[BẢNG 1] SỐ LƯỢNG INSTANCES (Đối tượng) CỦA TỪNG LỚP")
    print(header)
    print("-" * len(header))

    for cls in all_classes:
        tr = instance_stats['train'][cls]
        va = instance_stats['valid'][cls]
        te = instance_stats['test'][cls]
        total = tr + va + te
        print(f"{cls:<10} | {tr:<10} | {va:<10} | {te:<10} | {total:<10}")

    print("-" * len(header))

    # ---------------------------------------------------------
    # BẢNG 2: IN BẢNG SỐ LƯỢNG ẢNH
    # ---------------------------------------------------------
    print("\n[BẢNG 2] SỐ LƯỢNG ẢNH CÓ CHỨA TỪNG LỚP")
    print(header)
    print("-" * len(header))

    for cls in all_classes:
        tr = image_stats['train'][cls]
        va = image_stats['valid'][cls]
        te = image_stats['test'][cls]
        total = tr + va + te
        print(f"{cls:<10} | {tr:<10} | {va:<10} | {te:<10} | {total:<10}")

    print("-" * len(header))

    # ---------------------------------------------------------
    # TỔNG SỐ ẢNH THỰC TẾ
    # ---------------------------------------------------------
    print("\nTổng số lượng file ảnh thực tế trong các thư mục:")
    for split in splits:
        img_dir = os.path.join(split_root, split, 'images')
        if os.path.exists(img_dir):
            num_imgs = len(list(Path(img_dir).glob('*')))
            print(f" └─ Tập {split:<5}: {num_imgs} ảnh")

if __name__ == "__main__":
    split_data_path = '/content/Dataset_split'
    count_labels_in_splits(split_data_path)
