import cv2
import numpy as np

def encode_iris(normalized_img, mask=None):
    """
    Tao IrisCode su dung Gabor phuc (0 do).
    Dam bao kernel la zero-mean de khong bi anh huong boi do sang.
    """
    img_float = normalized_img.astype(np.float32)
    img_float -= np.mean(img_float)
    
    ksize = 15
    sigma = 4.0
    theta = 0
    lambd = 10.0
    gamma = 0.5
    
    kernel_real = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
    kernel_imag = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, np.pi/2, ktype=cv2.CV_32F)
    
    # Ép kernel ve zero-mean
    kernel_real -= np.mean(kernel_real)
    kernel_imag -= np.mean(kernel_imag)
    
    f_real = cv2.filter2D(img_float, -1, kernel_real)
    f_imag = cv2.filter2D(img_float, -1, kernel_imag)
    
    # Dung 2 bits (thuc va ao) cho moi pixel
    code1 = (f_real > 0).astype(np.uint8)
    code2 = (f_imag > 0).astype(np.uint8)
    
    iris_code = np.concatenate([code1, code2], axis=1)
    
    if mask is None:
        mask = np.ones_like(normalized_img, dtype=np.uint8)
    else:
        mask = (mask > 0).astype(np.uint8)
        
    final_mask = np.concatenate([mask, mask], axis=1)
        
    return iris_code, final_mask
