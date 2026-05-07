import pytest
import numpy as np
from src.core.normalization import normalize_iris

def test_normalize_iris_shape():
    """Kiem tra kich thuoc anh sau khi chuan hoa."""
    # Anh gia lap 400x400
    img = np.zeros((400, 400), dtype=np.uint8)
    pupil = (200, 200, 50)
    iris = (200, 200, 150)
    
    norm_img = normalize_iris(img, pupil, iris)
    
    # Kich thuoc mac dinh trong plan la 64x512
    assert norm_img.shape == (64, 512)
    assert isinstance(norm_img, np.ndarray)

def test_normalize_iris_values():
    """Kiem tra gia tri pixel sau khi unwrap (don gian)."""
    # Tao mot hinh xuyen co mau trang (255) tren nen den (0)
    img = np.zeros((400, 400), dtype=np.uint8)
    cv2_img = cv2.circle(img, (200, 200), 100, 255, -1) # Ve iris trang
    cv2_img = cv2.circle(cv2_img, (200, 200), 50, 0, -1) # Ve dong tu den
    
    pupil = (200, 200, 50)
    iris = (200, 200, 150)
    
    # normalize_iris se lay vung giua 50 va 150
    # Vung tu 50 den 100 se la trang, tu 100 den 150 se la den
    norm_img = normalize_iris(cv2_img, pupil, iris)
    
    # Kiem tra mot vai diem
    assert norm_img[10, 0] == 255 # Vung gan dong tu (trang)
    assert norm_img[50, 0] == 0   # Vung gan ria ngoai (den)

import cv2 # Import de dung trong test_normalize_iris_values
