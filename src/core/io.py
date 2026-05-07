import cv2
import os
import kagglehub

def load_image(image_path: str = None, subject_id: str = None, side: str = "L", sequence: str = "00"):
    """
    Doc anh mong mat tu duong dan hoac tu ID cua dataset.
    
    Tham so:
    - image_path: Duong dan truc tiep (tuy chon).
    - subject_id: ID doi tuong (vd: '000').
    - side: Mat trai/phai ('L' hoac 'R').
    - sequence: Thu tu anh ('00' - '09').
    """
    if image_path is None:
        if subject_id is None:
            raise ValueError("Phai cung cap image_path hoac subject_id")
            
        # Su dung kagglehub de lay thu muc dataset
        path = kagglehub.dataset_download("sondosaabed/casia-iris-thousand")
        # CASIA-Iris-Thousand structure: .../CASIA-Iris-Thousand/CASIA-Iris-Thousand/<subject>/<side>/S5<subject><side><seq>.jpg
        # Note: Based on previous experience, it might be nested
        base_dir = os.path.join(path, "CASIA-Iris-Thousand")
        filename = f"S5{subject_id}{side}{sequence}.jpg"
        image_path = os.path.join(base_dir, subject_id, side, filename)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Khong tim thay anh tai: {image_path}")

    # Doc anh o che do grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"File khong phai la anh hop le: {image_path}")
        
    # Ap dung CLAHE de tang cuong do tuong phan cho texture mong mat
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img)
    
    return img_enhanced
