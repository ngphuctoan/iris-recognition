import cv2
import numpy as np

import cv2
import numpy as np

def encode_iris(normalized_img, mask=None):
    """
    Tạo IrisCode sử dụng bộ lọc Gabor (Phép biến đổi pha).
    
    Quy trình:
    1. Áp dụng bộ lọc Gabor (phần thực và phần ảo) lên ảnh đã chuẩn hóa.
    2. Lấy dấu của kết quả phản hồi (Pha) để chuyển sang dạng nhị phân (0 hoặc 1).
    3. Kết quả là một mã IrisCode nhị phân cực kỳ nhẹ và dễ so khớp.
    """
    img_float = normalized_img.astype(np.float32)
    # Loại bỏ giá trị trung bình để giảm ảnh hưởng của độ sáng toàn cục
    img_float -= np.mean(img_float)
    
    # Cấu hình bộ lọc Gabor
    ksize = 15
    sigma = 4.0
    theta = 0
    lambd = 10.0
    gamma = 0.5
    
    # Tạo nhân (Kernel) Gabor phần thực và phần ảo
    kernel_real = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
    kernel_imag = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, np.pi/2, ktype=cv2.CV_32F)
    
    # Ép kernel về zero-mean để lọc bỏ nhiễu DC
    kernel_real -= np.mean(kernel_real)
    kernel_imag -= np.mean(kernel_imag)
    
    # Thực hiện tích chập (Convolution) ảnh với bộ lọc
    f_real = cv2.filter2D(img_float, -1, kernel_real)
    f_imag = cv2.filter2D(img_float, -1, kernel_imag)
    
    # Lượng tử hóa pha: Nếu phản hồi > 0 thì bit là 1, ngược lại là 0
    code1 = (f_real > 0).astype(np.uint8)
    code2 = (f_imag > 0).astype(np.uint8)
    
    # Kết hợp 2 dải bit (thực và ảo) để tạo mã IrisCode cuối cùng
    iris_code = np.concatenate([code1, code2], axis=1)
    
    if mask is None:
        mask = np.ones_like(normalized_img, dtype=np.uint8)
    else:
        # Chuyển mặt nạ sang dạng nhị phân
        mask = (mask > 0).astype(np.uint8)
        
    # Mặt nạ cũng phải được nhân đôi để khớp với kích thước của IrisCode
    final_mask = np.concatenate([mask, mask], axis=1)
        
    return iris_code, final_mask
