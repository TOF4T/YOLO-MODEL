import os
import shutil
import argparse
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def build_stratification_dataframe(img_dir, txt_dir):
    """
    Bước 1: Quét dữ liệu và xây dựng DataFrame chứa các đặc trưng bounding box.
    """
    print(f"1. Đang đọc dữ liệu từ: {img_dir} và {txt_dir}...")
    valid_exts = ('.jpg', '.jpeg', '.png')
    
    # Lấy danh sách file ảnh, bỏ qua nếu thư mục không tồn tại
    if not os.path.exists(img_dir):
        return pd.DataFrame()
        
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)]
    
    if not img_files:
        return pd.DataFrame()
    
    l_data = []
    for img_filename in img_files:
        fname = os.path.splitext(img_filename)[0]
        txt_path = os.path.join(txt_dir, fname + '.txt')
        
        if not os.path.exists(txt_path):
            # Ảnh nền (Background)
            l_data.append([img_filename, -1, np.nan, np.nan, np.nan, np.nan])
        else:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            if not lines:
                l_data.append([img_filename, -1, np.nan, np.nan, np.nan, np.nan])
            else:
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        l_data.append([img_filename, cls, x, y, w, h])

    df = pd.DataFrame(l_data, columns=['filename', 'class', 'x', 'y', 'w', 'h'])
    
    # One-hot encoding và loại bỏ cột class -1
    one_hot = pd.get_dummies(df['class'], prefix='class')
    if 'class_-1' in one_hot.columns:
        one_hot = one_hot.drop(columns=['class_-1'])

    data = pd.concat([df, one_hot], axis=1)
    data['w'] = data['w'] * 1000
    data['h'] = data['h'] * 1000

    data = data.drop(columns=['class', 'x', 'y'])
    grouped_df = data.groupby('filename').sum().reset_index()

    class_cols = [col for col in grouped_df.columns if col.startswith('class_')]
    grouped_df['box_count'] = grouped_df[class_cols].sum(axis=1).replace(0, 1)

    grouped_df['avg_w'] = grouped_df['w'] / grouped_df['box_count']
    grouped_df['avg_h'] = grouped_df['h'] / grouped_df['box_count']
    grouped_df['avg_ratio'] = grouped_df['avg_h'] / (grouped_df['avg_w'] + 1e-6) 

    new_df = grouped_df.drop(columns=['w', 'h', 'box_count']).fillna(0)
    return new_df

def split_dataset(new_df):
    """
    Bước 2: Phân tách 2 giai đoạn (Tách tập Test -> Tách tập Train/Valid).
    """
    print("\n2. Bắt đầu phân tầng đa nhãn 2 giai đoạn...")
    X = new_df[['filename']].values
    Y = new_df.drop(columns=['filename']).values

    # Giai đoạn 1: Lấy tập Test cố định (10-fold)
    mskf_test = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    stage1_splits = list(mskf_test.split(X, Y))
    rest_index, test_index = stage1_splits[-1] 
    
    test_files = X[test_index].flatten().tolist()
    print(f"   -> Đã trích xuất {len(test_files)} ảnh làm tập Test cố định.")

    # Giai đoạn 2: Lấy 1 cặp Train/Valid từ dữ liệu còn lại (9-fold)
    X_rest, Y_rest = X[rest_index], Y[rest_index]
    mskf_train = MultilabelStratifiedKFold(n_splits=9, shuffle=True, random_state=42)
    
    train_idx, val_idx = next(mskf_train.split(X_rest, Y_rest))
    train_files = X_rest[train_idx].flatten().tolist()
    val_files = X_rest[val_idx].flatten().tolist()
    
    print(f"   -> Đã chia Train: {len(train_files)} ảnh | Valid: {len(val_files)} ảnh.")
    return train_files, val_files, test_files

def copy_to_yolo_structure(img_dir, txt_dir, output_dir, train_files, val_files, test_files):
    """
    Bước 3: Tự động copy file vào các thư mục theo format yêu cầu.
    """
    print("\n3. Đang tổ chức cấu trúc thư mục...")
    folders = ['train', 'valid', 'test']
    
    # Tạo cấu trúc: output_dir/{split}/images và output_dir/{split}/labels
    for folder in folders:
        os.makedirs(os.path.join(output_dir, folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, folder, 'labels'), exist_ok=True)

    file_mapping = {
        'train': train_files,
        'valid': val_files,
        'test': test_files
    }

    for split_name, files in file_mapping.items():
        for fname in files:
            # Copy ảnh
            src_img = os.path.join(img_dir, fname)
            dst_img = os.path.join(output_dir, split_name, 'images', fname)
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
            
            # Copy nhãn
            txt_name = os.path.splitext(fname)[0] + '.txt'
            src_txt = os.path.join(txt_dir, txt_name)
            dst_txt = os.path.join(output_dir, split_name, 'labels', txt_name)
            if os.path.exists(src_txt):
                shutil.copy(src_txt, dst_txt)
                
    print(f"HOÀN TẤT! Dữ liệu đã sẵn sàng tại thư mục: {output_dir}")

# ==========================================
# THỰC THI CHÍNH
# ==========================================
if __name__ == "__main__":
    # Cấu hình đọc tham số từ dòng lệnh
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='.', help='Đường dẫn tới thư mục gốc chứa thư mục images và labels')
    args = parser.parse_args()

    # Thiết lập đường dẫn dựa trên tham số truyền vào
    RAW_IMG_DIR = os.path.join(args.datapath, "images")
    RAW_TXT_DIR = os.path.join(args.datapath, "labels")
    OUTPUT_YOLO_DIR = os.path.join(args.datapath, "YOLO11_Dataset")
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(RAW_IMG_DIR, exist_ok=True)
    os.makedirs(RAW_TXT_DIR, exist_ok=True)

    # Chạy quy trình
    df_features = build_stratification_dataframe(RAW_IMG_DIR, RAW_TXT_DIR)
    
    if not df_features.empty:
        train_list, val_list, test_list = split_dataset(df_features)
        copy_to_yolo_structure(RAW_IMG_DIR, RAW_TXT_DIR, OUTPUT_YOLO_DIR, train_list, val_list, test_list)
    else:
        print(f"\n[LỖI] Không tìm thấy dữ liệu ảnh/nhãn trong thư mục: {RAW_IMG_DIR}")
        print("Vui lòng kiểm tra lại xem bạn đã giải nén dữ liệu vào đúng thư mục datapath chưa.")
