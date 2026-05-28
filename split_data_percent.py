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
    - Tính chuẩn tỷ lệ: 56% Train, 14% Valid, 30% Test.
    - Đảm bảo Test >= 3 và Valid >= 2 sớm nhất có thể (từ n >= 8).
    - Bảo vệ tập Train luôn là tập có nhiều dữ liệu nhất.
    """
    # 1. Bố trí thủ công: Ưu tiên chạm mốc Valid=2, Test=3 từ n=8
    if n <= 0: return 0, 0, 0
    if n == 1: return 1, 0, 0 
    if n == 2: return 1, 0, 1 
    if n == 3: return 1, 1, 1 
    if n == 4: return 2, 1, 1
    if n == 5: return 2, 1, 2  
    if n == 6: return 3, 1, 2
    if n == 7: return 3, 2, 2
    if n == 8: return 3, 2, 3  # Chạm mốc Valid=2, Test=3
    if n == 9: return 4, 2, 3  
    if n == 10: return 5, 2, 3 

    # 2. Xử lý tỷ lệ cho n >= 11 (Tự động đạt >= 3 cho Test và >= 2 cho Valid)
    c_train = int(round(n * 0.56))
    c_valid = int(round(n * 0.14))
    c_test = int(round(n * 0.30))

    # Điều chỉnh nếu tổng bị lệch do sai số làm tròn
    current_total = c_train + c_valid + c_test
    
    while current_total > n:
        if c_test > int(n * 0.30): c_test -= 1
        elif c_valid > int(n * 0.14): c_valid -= 1
        else: c_train -= 1
        current_total = c_train + c_valid + c_test
        
    while current_total < n:
        c_train += 1
        current_total = c_train + c_valid + c_test

    # 3. Thuật toán mượn ảnh an toàn (Dự phòng
