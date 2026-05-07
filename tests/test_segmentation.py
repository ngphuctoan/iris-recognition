import pytest
import numpy as np
import cv2
from src.core.segmentation import segment_iris
from src.core.io import load_image
from pathlib import Path

def test_segment_iris_real_image():
    """Kiem tra tach mong mat tren anh that tu dataset."""
    # Lấy một ảnh mẫu từ subset đã chuẩn bị
    subset_dir = Path("data/subset")
    sample_file = next(subset_dir.glob("**/S5*.jpg"))
    
    # Parse filename: S5<subject><side><seq>.jpg
    # Ví dụ: S5000L00.jpg -> subject='000', side='L', seq='00'
    fname = sample_file.name
    subject_id = fname[2:5]
    side = fname[5]
    seq = fname[6:8]
    
    img = load_image(dataset_dir=str(subset_dir), subject_id=subject_id, side=side, sequence=seq)
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
