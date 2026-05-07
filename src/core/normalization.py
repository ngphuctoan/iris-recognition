import numpy as np
import cv2

def normalize_iris(img, pupil, iris, res_r=64, res_theta=512, mask=None):
    """
    Chuan hoa vung mong mat sang hinh chu nhat (Rubber Sheet Model).
    
    Tham so:
    - img: Anh grayscale goc.
    - pupil: (xp, yp, rp) - thong so dong tu.
    - iris: (xi, yi, ri) - thong so mong mat.
    - res_r: Do phan giai chieu doc (ban kinh).
    - res_theta: Do phan giai chieu ngang (goc).
    - mask: Mat na nhiễu (tùy chọn).
    
    Tra ve:
    - np.ndarray hoac tuple: Anh da duoc unwrap, va mask neu co.
    """
    xp, yp, rp = pupil
    xi, yi, ri = iris
    
    r_vals = np.linspace(0, 1, res_r)
    theta_vals = np.linspace(0, 2 * np.pi, res_theta)
    
    r_grid, theta_grid = np.meshgrid(r_vals, theta_vals, indexing='ij')
    
    x_p_edge = xp + rp * np.cos(theta_grid)
    y_p_edge = yp + rp * np.sin(theta_grid)
    x_i_edge = xi + ri * np.cos(theta_grid)
    y_i_edge = yi + ri * np.sin(theta_grid)
    
    x_map = (1 - r_grid) * x_p_edge + r_grid * x_i_edge
    y_map = (1 - r_grid) * y_p_edge + r_grid * y_i_edge
    
    # Su dung cv2.remap de co chat luong anh tot hon (bilinear interpolation)
    normalized_img = cv2.remap(img, x_map.astype(np.float32), y_map.astype(np.float32), cv2.INTER_LINEAR)
    
    if mask is not None:
        normalized_mask = cv2.remap(mask, x_map.astype(np.float32), y_map.astype(np.float32), cv2.INTER_NEAREST)
        return normalized_img.astype(np.uint8), normalized_mask.astype(np.uint8)
    
    return normalized_img.astype(np.uint8)
