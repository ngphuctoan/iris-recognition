import pytest
import numpy as np
from src.core.encoding import encode_iris
from src.core.matching import match_codes, hamming_distance

def test_encode_iris_output():
    """Kiem tra ma IrisCode duoc tao ra."""
    # Anh da chuan hoa 64x512
    norm_img = np.random.randint(0, 255, (64, 512), dtype=np.uint8)
    
    code, mask = encode_iris(norm_img)
    
    # IrisCode va Mask phai cung kich thuoc voi anh dau vao
    assert code.shape == norm_img.shape
    assert mask.shape == norm_img.shape
    # Phai la mang nhị phân (0 hoac 1)
    assert np.all((code == 0) | (code == 1))

def test_hamming_distance_identity():
    """Kiem tra khoang cach Hamming giua 2 ma giong het nhau."""
    code = np.random.randint(0, 2, (64, 512), dtype=np.uint8)
    mask = np.ones((64, 512), dtype=np.uint8)
    
    dist = hamming_distance(code, code, mask, mask)
    assert dist == 0.0

def test_hamming_distance_different():
    """Kiem tra khoang cach Hamming giua 2 ma khac nhau."""
    code1 = np.zeros((64, 512), dtype=np.uint8)
    code2 = np.ones((64, 512), dtype=np.uint8)
    mask = np.ones((64, 512), dtype=np.uint8)
    
    dist = hamming_distance(code1, code2, mask, mask)
    assert dist == 1.0

def test_match_codes_threshold():
    """Kiem tra logic so khop (True/False) dua tren nguong."""
    # ... logic so khop ...
    pass
