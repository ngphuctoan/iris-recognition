import pytest
import numpy as np
import cv2
import os
from src.core.io import load_image

def test_load_image_success(tmp_path):
    """Kiểm tra đọc ảnh thành công từ cấu trúc dataset giả lập."""
    # Tạo cấu trúc dataset giả lập: tmp_path/000/S5000L00.jpg
    subject_dir = tmp_path / "000"
    subject_dir.mkdir()
    img_file = subject_dir / "S5000L00.jpg"
    
    test_img = np.zeros((100, 100), dtype=np.uint8)
    cv2.imwrite(str(img_file), test_img)
    
    # Load sử dụng dataset_dir và subject_id
    loaded_img = load_image(dataset_dir=str(tmp_path), subject_id="000", side="L", sequence="00")
    assert loaded_img is not None
    assert loaded_img.shape == (100, 100)
    assert isinstance(loaded_img, np.ndarray)

def test_load_image_not_found(tmp_path):
    """Kiểm tra lỗi khi không tìm thấy subject trong dataset_dir."""
    with pytest.raises(FileNotFoundError):
        load_image(dataset_dir=str(tmp_path), subject_id="999")

def test_load_image_invalid_format(tmp_path):
    """Kiểm tra lỗi khi tệp tin bị hỏng hoặc không đúng định dạng."""
    subject_dir = tmp_path / "001"
    subject_dir.mkdir()
    txt_file = subject_dir / "S5001L00.jpg" # Giả làm file ảnh
    
    with open(txt_file, "w") as f:
        f.write("đây không phải là dữ liệu ảnh")
    
    with pytest.raises(ValueError):
        load_image(dataset_dir=str(tmp_path), subject_id="001")
