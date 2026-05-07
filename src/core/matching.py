import numpy as np
import cv2

def hamming_distance(code1, code2, mask1, mask2):
    """Tinh khoang cach Hamming giua 2 ma IrisCode."""
    combined_mask = cv2.bitwise_and(mask1, mask2)
    xor_res = cv2.bitwise_xor(code1, code2)
    diff_bits = cv2.bitwise_and(xor_res, combined_mask)
    
    num_diff = np.sum(diff_bits)
    num_valid = np.sum(combined_mask)
    
    if num_valid == 0:
        return 1.0
        
    return float(num_diff / num_valid)

def match_codes(code1, code2, mask1, mask2, threshold=0.40):
    """
    So khop 2 ma voi xu ly xoay (Rotation handling).
    Moi pixel co 8 bits, nen dich chuyen phai theo boi so cua 8.
    """
    min_dist = 1.0
    # Thu dich chuyen tu -30 den 30 pixels (tuong ung -60 den 60 bits)
    for pixel_shift in range(-30, 31):
        shift = pixel_shift * 2
        shifted_code = np.roll(code2, shift, axis=1)
        shifted_mask = np.roll(mask2, shift, axis=1)
        
        dist = hamming_distance(code1, shifted_code, mask1, shifted_mask)
        if dist < min_dist:
            min_dist = dist
            
    return min_dist < threshold, min_dist
