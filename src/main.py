from src.core.io import load_image
from src.core.segmentation import segment_iris, detect_eyelines
from src.core.normalization import normalize_iris
from src.core.encoding import encode_iris
from src.core.matching import match_codes
from src.utils.visualization import show_segmentation, show_normalization, plot_debug_match
from pathlib import Path
import numpy as np

def demo():
    print("--- Iris Recognition System Demo ---")
    
    # 1. Chon anh mau
    subset_dir = Path("data/subset")
    if not subset_dir.exists():
        print(f"Error: Khong tim thay thu muc {subset_dir}. Chay tu goc thu muc project.")
        return
        
    subject_dirs = sorted([d for d in subset_dir.iterdir() if d.is_dir()])
    if not subject_dirs:
        print("Error: Dataset subset trong.")
        return
    
    # Lay 2 anh cua cung mot nguoi de so khop (identity match)
    subject = subject_dirs[0]
    imgs = sorted(list(subject.glob("*.jpg")))
    
    if len(imgs) < 2:
        print(f"Khong du anh de demo cho Subject {subject.name}!")
        return
    
    print(f"Dang xu ly Subject {subject.name}: {imgs[0].name} va {imgs[1].name}")
    
    # Pipeline cho ảnh 1
    fname1 = imgs[0].name
    img1 = load_image(subject_id=fname1[2:5], side=fname1[5], sequence=fname1[6:8])
    p1, i1 = segment_iris(img1)
    mask_raw1 = detect_eyelines(img1, p1, i1)
    norm1, norm1_mask = normalize_iris(img1, p1, i1, mask=mask_raw1)
    code1, mask1 = encode_iris(norm1, mask=norm1_mask)
    
    # Pipeline cho ảnh 2
    fname2 = imgs[1].name
    img2 = load_image(subject_id=fname2[2:5], side=fname2[5], sequence=fname2[6:8])
    p2, i2 = segment_iris(img2)
    mask_raw2 = detect_eyelines(img2, p2, i2)
    norm2, norm2_mask = normalize_iris(img2, p2, i2, mask=mask_raw2)
    code2, mask2 = encode_iris(norm2, mask=norm2_mask)
    
    # So khop
    is_match, distance = match_codes(code1, code2, mask1, mask2)
    
    print(f"Ket qua so khop: {'GIONG NHAU' if is_match else 'KHAC NHAU'}")
    print(f"Khoang cach Hamming: {distance:.4f}")
    
    # Hien thi debug
    print("Demo hoan tat. Pipeline sinh trac hoc hoat dong tot.")

if __name__ == "__main__":
    demo()
