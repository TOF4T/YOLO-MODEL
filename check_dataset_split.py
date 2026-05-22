import os
from pathlib import Path
from collections import Counter

def count_labels_in_splits(split_root):
    """
    Quét qua các thư mục train, val, test và đếm số lượng instance của từng class
    """
    splits = ['train', 'valid', 'test']

    stats = {split: Counter() for split in splits}

    print(f"🔍 Đang kiểm tra dữ liệu tại: {split_root}")
    print("-" * 50)

    for split in splits:
        label_dir = os.path.join(split_root, split, 'labels')

        if not os.path.exists(label_dir):
            print(f"⚠️ Cảnh báo: Không tìm thấy thư mục nhãn cho tập {split} tại {label_dir}")
            continue

        label_files = list(Path(label_dir).glob('*.txt'))

        for lbl in label_files:
            with open(lbl, 'r') as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        class_id = str(int(parts[0]))
                        stats[split][class_id] += 1

    all_classes = sorted(list(set().union(*(s.keys() for s in stats.values()))))

    header = f"{'Class ID':<10} | {'Train':<10} | {'Valid':<10} | {'Test':<10} | {'Total':<10}"
    print(header)
    print("-" * len(header))

    for cls in all_classes:
        tr = stats['train'][cls]
        va = stats['valid'][cls]
        te = stats['test'][cls]
        total = tr + va + te
        print(f"{cls:<10} | {tr:<10} | {va:<10} | {te:<10} | {total:<10}")

    print("-" * len(header))
    print("Tổng số lượng file ảnh thực tế:")
    for split in splits:
        img_dir = os.path.join(split_root, split, 'images')
        if os.path.exists(img_dir):
            num_imgs = len(list(Path(img_dir).glob('*')))
            print(f" └─ Tập {split:<5}: {num_imgs} ảnh")

if __name__ == "__main__":
    split_data_path = '/content/Dataset_split'
    count_labels_in_splits(split_data_path)
