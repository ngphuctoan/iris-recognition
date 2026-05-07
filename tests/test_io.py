import pytest
import numpy as np
import cv2
import os
from src.core.io import load_image

def test_load_image_success(tmp_path):
    """Kiem tra doc anh thanh cong."""
    # Tao mot anh gia lap
    img_path = str(tmp_path / "test.jpg")
    test_img = np.zeros((100, 100), dtype=np.uint8)
    cv2.imwrite(img_path, test_img)
    
    loaded_img = load_image(img_path)
    assert loaded_img is not None
    assert loaded_img.shape == (100, 100)
    assert isinstance(loaded_img, np.ndarray)

def test_load_image_not_found():
    """Kiem tra loi khi khong tim thay file."""
    with pytest.raises(FileNotFoundError):
        load_image("non_existent_image.jpg")

def test_load_image_invalid_format(tmp_path):
    """Kiem tra loi khi file khong phai la anh hop le."""
    txt_path = str(tmp_path / "test.txt")
    with open(txt_path, "w") as f:
        f.write("day khong phai la anh")
    
    with pytest.raises(ValueError):
        load_image(txt_path)
