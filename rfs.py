import os
import shutil
import random
import math
from pathlib import Path
from collections import Counter

def apply_repeat_factor_sampling(split_root, t=0.1, reduction='max'):
    """
    Áp dụng Repeat-Factor Sampling trực tiếp bằng cách nhân bản file trong tập train.
    - t: Ngưỡng kiểm soát việc lấy mẫu dư (ví dụ: 0.1 nghĩa là lớp nào xuất hiện dưới 10% số ảnh sẽ được oversample).
    - reduction: Cách thu gọn hệ số khi ảnh có nhiều lớp ('max' hoặc 'mean').
    """
    train_dir = os.path.join(split_root, 'train')
    img_dir = os.path.join(train_dir, 'images')
    lbl_dir = os.path.join(train_dir, 'labels')

    # 1. Đếm tần suất xuất hiện f_c của từng class (tính theo số lượng ảnh chứa class đó)
    lbl_files = list(Path(lbl_dir).glob('*.txt'))
    total_images = len(lbl_files)
    if total_images == 0:
        print(" Không tìm thấy file nhãn nào trong tập huấn luyện!")
        return

    class_image_counts = Counter()
    image_to_classes = {}

    for lbl in lbl_files:
        with open(lbl, 'r') as f:
            classes_in_file = set()
            for line in f:
                parts = line.split()
                if parts:
                    # YOLO format: class_id x_center y_center width height
                    classes_in_file.add(int(parts[0]))
        image_to_classes[lbl.stem] = classes_in_file
        for c in classes_in_file:
            class_image_counts[c] += 1

    # Tính f_c = số ảnh chứa class c / tổng số ảnh train
    f_c = {c: count / total_images for c, count in class_image_counts.items()}

    # 2. Tính hệ số lặp r_c cho từng class theo công thức chuẩn RFS toán học
    # r_c = max(1, sqrt(t / f_c))
    r_c = {c: max(1.0, math.sqrt(t / freq)) for c, freq in f_c.items()}

    print(" --- Tần suất (f_c) và Hệ số lặp (r_c) từng lớp ---")
    for c in sorted(r_c.keys()):
        print(f"Class {c}: f_c = {f_c[c]:.4f} -> r_c = {r_c[c]:.2f}")

    # 3. Tính r_i cho từng bức ảnh và tiến hành nhân bản file
    print("\n Đang tiến hành nhân bản file (Oversampling cho các lớp đuôi dài)...")
    copied_count = 0
    valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

    for lbl_path in lbl_files:
        img_name_base = lbl_path.stem
        classes_in_image = image_to_classes[img_name_base]

        if not classes_in_image:
            continue

        # Lấy danh sách r_c của các class có mặt trong bức ảnh này
        r_constants = [r_c[c] for c in classes_in_image]

        # Áp dụng cơ chế rút gọn reduction ('max' hoặc 'mean')
        if reduction == 'max':
            r_i = max(r_constants)
        elif reduction == 'mean':
            r_i = sum(r_constants) / len(r_constants)
        else:
            r_i = max(r_constants)

        # Xử lý phần thập phân bằng xác suất ngẫu nhiên
        rep_factor = int(r_i)
        if random.random() < (r_i - rep_factor):
            rep_factor += 1

        # Số lượng bản sao cần tạo thêm (rep_factor - 1)
        num_copies = rep_factor - 1

        if num_copies > 0:
            # Tìm file ảnh gốc tương ứng với file txt nhãn
            img_path = None
            for ext in valid_extensions:
                potential_path = os.path.join(img_dir, img_name_base + ext)
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break

            if not img_path:
                continue # Bỏ qua nếu ảnh bị lệch hoặc thiếu file ảnh gốc

            # Sao chép file nhãn và ảnh ra cùng một thư mục với hậu tố phân biệt
            for k in range(num_copies):
                suffix = f"_rfs_{k+1}"
                new_img_path = os.path.join(img_dir, f"{img_name_base}{suffix}{Path(img_path).suffix}")
                new_lbl_path = os.path.join(lbl_dir, f"{img_name_base}{suffix}.txt")

                shutil.copy(img_path, new_img_path)
                shutil.copy(lbl_path, new_lbl_path)
                copied_count += 1

    print(f" Hoàn thành! Đã tạo thêm {copied_count} bản sao cho các ảnh chứa tail classes.")

# Thực thi hàm áp dụng trực tiếp lên tập dữ liệu đã split của bạn
# Bạn có thể tinh chỉnh t (0.05, 0.1, 0.2) tùy mức độ mất cân bằng dữ liệu của bạn
apply_repeat_factor_sampling('/content/Dataset_split', t=0.1, reduction='max')
