import numpy as np
import cv2

import numpy as np
import cv2

def hamming_distance(code1, code2, mask1, mask2):
    """
    Tính khoảng cách Hamming giữa 2 mã IrisCode.
    
    Quy trình:
    1. Kết hợp 2 mặt nạ nhiễu (AND) để chỉ so sánh các vùng hợp lệ.
    2. Sử dụng XOR để tìm các bit khác biệt giữa 2 mã.
    3. Tỷ lệ giữa số bit khác biệt và tổng số bit hợp lệ chính là khoảng cách Hamming.
    """
    # Chỉ so sánh các bit mà cả 2 ảnh đều không bị nhiễu (mí mắt, lông mi)
    combined_mask = cv2.bitwise_and(mask1, mask2)
    
    # Tìm các bit khác nhau (1 XOR 0 = 1, 0 XOR 0 = 0)
    xor_res = cv2.bitwise_xor(code1, code2)
    
    # Chỉ giữ lại các bit khác biệt nằm trong vùng hợp lệ
    diff_bits = cv2.bitwise_and(xor_res, combined_mask)
    
    num_diff = np.sum(diff_bits)
    num_valid = np.sum(combined_mask)
    
    if num_valid == 0:
        return 1.0 # Nếu không có vùng nào hợp lệ, coi như không khớp
        
    return float(num_diff / num_valid)

def match_codes(code1, code2, mask1, mask2, threshold=0.40):
    """
    So khớp 2 mã IrisCode với xử lý xoay (Rotation handling).
    Do mắt người có thể bị nghiêng khi chụp, ta phải thử dịch chuyển mã theo chiều ngang.
    """
    min_dist = 1.0
    
    # Thử dịch chuyển từ -30 đến 30 pixels để bù đắp cho việc nghiêng đầu
    for pixel_shift in range(-30, 31):
        # Mỗi bước dịch chuyển tương ứng với việc cuộn (roll) mảng nhị phân
        shift = pixel_shift * 2
        shifted_code = np.roll(code2, shift, axis=1)
        shifted_mask = np.roll(mask2, shift, axis=1)
        
        dist = hamming_distance(code1, shifted_code, mask1, shifted_mask)
        if dist < min_dist:
            min_dist = dist
            
    # Nếu khoảng cách nhỏ nhất thấp hơn ngưỡng (threshold), ta coi là khớp
    return min_dist < threshold, min_dist
