import pytest
import numpy as np
import cv2
from src.core.segmentation import segment_iris
from src.core.io import load_image
from pathlib import Path

def test_segment_iris_real_image():
    """Kiem tra tach mong mat tren anh that tu dataset."""
    # Lay mot anh mau tu subset da chuan bi
    subset_dir = Path("/home/muffin/Desktop/University/Information-Security/iris-recognition/data/subset")
    sample_img_path = next(subset_dir.glob("**/*.jpg"))
    
    img = load_image(str(sample_img_path))
    pupil, iris = segment_iris(img)
    
    # Kiem tra ket qua tra ve co dung dinh dang (x, y, r)
    assert len(pupil) == 3
    assert len(iris) == 3
    
    # Dong tu phai nho hon mong mat
    assert pupil[2] < iris[2]
    
    # Tam cua chung phai gan nhau (dong tam)
    dist = np.sqrt((pupil[0] - iris[0])**2 + (pupil[1] - iris[1])**2)
    assert dist < 50 # Cho phep sai so nho

def test_segment_iris_invalid_input():
    """Kiem tra loi khi dau vao khong hop le."""
    with pytest.raises(TypeError):
        segment_iris(None)
