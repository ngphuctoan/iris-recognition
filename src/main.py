from src.core.io import load_image
from src.core.segmentation import segment_iris
from src.core.normalization import normalize_iris
from src.core.encoding import encode_iris
from src.core.matching import match_codes
from src.utils.visualization import show_segmentation, show_normalization, plot_debug_match
from pathlib import Path

def demo():
    print("--- Iris Recognition System Demo ---")
    
    # 1. Chon anh mau
    subset_dir = Path("data/subset")
    subject_dirs = sorted([d for d in subset_dir.iterdir() if d.is_dir()])
    
    # Lay 2 anh cua cung mot nguoi de so khop (identity match)
    subject = subject_dirs[0]
    imgs = sorted(list(subject.glob("*.jpg")))
    
    if len(imgs) < 2:
        print("Khong du anh de demo!")
        return
        
    img1_path = str(imgs[0])
    img2_path = str(imgs[1])
    
    print(f"Dang xu ly Subject {subject.name}: {imgs[0].name} va {imgs[1].name}")
    
    # Pipeline cho anh 1
    img1 = load_image(img1_path)
    p1, i1 = segment_iris(img1)
    norm1 = normalize_iris(img1, p1, i1)
    code1, mask1 = encode_iris(norm1)
    
    # Pipeline cho anh 2
    img2 = load_image(img2_path)
    p2, i2 = segment_iris(img2)
    norm2 = normalize_iris(img2, p2, i2)
    code2, mask2 = encode_iris(norm2)
    
    # So khop
    is_match, distance = match_codes(code1, code2, mask1, mask2)
    
    print(f"Ket qua so khop: {'GIONG NHAU' if is_match else 'KHAC NHAU'}")
    print(f"Khoang cach Hamming: {distance:.4f}")
    
    # Hien thi debug (Luu y: Matplotlib can GUI hoac luu file)
    print("Dang hien thi ket qua...")
    show_segmentation(img1, p1, i1, title=f"Segmentation Subject {subject.name}")
    show_normalization(norm1, title="Normalized Iris")
    plot_debug_match(code1, code2, mask1, mask2, distance)

if __name__ == "__main__":
    demo()
