import numpy as np
import cv2

def normalize_iris(img, pupil, iris, res_r=64, res_theta=512, mask=None):
    """
    Chuẩn hóa vùng mống mắt sang hình chữ nhật (Rubber Sheet Model của Daugman).
    
    Quy trình:
    1. Chuyển đổi tọa độ từ Cartesian (x,y) sang Polar (r, theta).
    2. Trải phẳng mống mắt thành một dải hình chữ nhật cố định.
    
    Tham số:
    - img: Ảnh grayscale gốc.
    - pupil: (xp, yp, rp) - thông số đồng tử.
    - iris: (xi, yi, ri) - thông số mống mắt.
    - res_r: Độ phân giải chiều dọc (radial).
    - res_theta: Độ phân giải chiều ngang (angular).
    - mask: Mặt nạ nhiễu (eyelid/eyelash).
    """
    xp, yp, rp = pupil
    xi, yi, ri = iris
    
    # Tạo lưới tọa độ cho mống mắt chuẩn hóa
    r_vals = np.linspace(0, 1, res_r)
    theta_vals = np.linspace(0, 2 * np.pi, res_theta)
    r_grid, theta_grid = np.meshgrid(r_vals, theta_vals, indexing='ij')
    
    # Tính toán các điểm biên của đồng tử và mống mắt tại từng góc theta
    x_p_edge = xp + rp * np.cos(theta_grid)
    y_p_edge = yp + rp * np.sin(theta_grid)
    x_i_edge = xi + ri * np.cos(theta_grid)
    y_i_edge = yi + ri * np.sin(theta_grid)
    
    # Nội suy tuyến tính để tìm tọa độ (x,y) tương ứng trong ảnh gốc
    x_map = (1 - r_grid) * x_p_edge + r_grid * x_i_edge
    y_map = (1 - r_grid) * y_p_edge + r_grid * y_i_edge
    
    # Sử dụng cv2.remap để lấy giá trị pixel từ ảnh gốc (Bilinear Interpolation)
    normalized_img = cv2.remap(img, x_map.astype(np.float32), y_map.astype(np.float32), cv2.INTER_LINEAR)
    
    # Nếu có mặt nạ nhiễu, ta cũng chuẩn hóa mặt nạ đó theo cùng một cách
    if mask is not None:
        # Với mặt nạ, ta dùng INTER_NEAREST để giữ giá trị nhị phân
        normalized_mask = cv2.remap(mask, x_map.astype(np.float32), y_map.astype(np.float32), cv2.INTER_NEAREST)
        return normalized_img.astype(np.uint8), normalized_mask.astype(np.uint8)
    
    return normalized_img.astype(np.uint8)
