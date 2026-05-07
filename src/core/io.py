import cv2
import os
import kagglehub

import cv2
import os
import kagglehub

def load_image(dataset_dir: str = "data/subset", subject_id: str = None, side: str = "L", sequence: str = "00"):
    """
    Đọc ảnh mống mắt từ thư mục dataset dựa trên metadata.
    Hỗ trợ cả cấu trúc thư mục phẳng (flat) và phân cấp (nested).
    
    Tham số:
    - dataset_dir: Thư mục chứa dataset (mặc định là data/subset).
    - subject_id: ID đối tượng (vd: '000').
    - side: Mắt trái/phải ('L' hoặc 'R').
    - sequence: Thứ tự ảnh ('00' - '09').
    """
    if subject_id is None:
        raise ValueError("Cần cung cấp subject_id để tải ảnh")
            
    filename = f"S5{subject_id}{side}{sequence}.jpg"
    image_path = None
    
    # 1. Thử tìm trong dataset_dir cục bộ trước (Ưu tiên tuyệt đối)
    if os.path.exists(dataset_dir):
        # Thử cấu trúc phẳng: <dataset_dir>/<subject>/<filename>
        path_flat = os.path.join(dataset_dir, subject_id, filename)
        # Thử cấu trúc phân cấp: <dataset_dir>/<subject>/<side>/<filename>
        path_nested = os.path.join(dataset_dir, subject_id, side, filename)
        
        if os.path.exists(path_flat):
            image_path = path_flat
        elif os.path.exists(path_nested):
            image_path = path_nested
    
    # 2. Nếu vẫn không thấy, thử dùng KaggleHub (dự phòng nếu mạng ổn định)
    if image_path is None:
        try:
            k_path = kagglehub.dataset_download("sondosaabed/casia-iris-thousand")
            # Kiểm tra các thư mục con phổ biến trong bản tải về của Kaggle
            for root_name in ["CASIA-Iris-Thousand", "2"]:
                p = os.path.join(k_path, root_name, subject_id, side, filename)
                if os.path.exists(p):
                    image_path = p
                    break
        except Exception:
            pass

    if image_path is None or not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy mẫu sinh trắc học cho Subject {subject_id} tại {dataset_dir}")

    # Đọc ảnh grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể giải mã dữ liệu ảnh tại: {image_path}")
        
    # Tăng cường độ tương phản (CLAHE) để làm nổi bật chi tiết mống mắt
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img)
