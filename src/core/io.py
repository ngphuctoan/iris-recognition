import cv2
import os

def load_image(image_path: str):
    """
    Doc anh grayscale tu duong dan.
    
    Tham so:
    - image_path: Duong dan den file anh.
    
    Tra ve:
    - np.ndarray: Anh duoi dang thang do xam (grayscale).
    
    Ngoai le:
    - FileNotFoundError: Neu file khong ton tai.
    - ValueError: Neu file khong phai la anh hop le.
    """
    # Kiem tra file ton tai
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Khong tim thay file tai: {image_path}")
    
    # Doc anh o che do grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"File khong phai la anh hop le: {image_path}")
        
    # Ap dung CLAHE de tang cuong do tuong phan cho texture mong mat
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img)
    
    return img_enhanced
