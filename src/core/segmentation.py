import cv2
import numpy as np

def segment_iris(img):
    """
    Tách đồng tử và mống mắt sử dụng Hough Circles và Adaptive Thresholding.
    
    Quy trình:
    1. Tiền xử lý (Làm mờ).
    2. Tìm đồng tử (Vùng tối nhất).
    3. Tìm mống mắt (Vùng bao quanh đồng tử).
    """
    if img is None:
        raise TypeError("Ảnh đầu vào không hợp lệ")
        
    # Bước 1: Làm mờ ảnh để giảm nhiễu, giúp tìm vòng tròn chính xác hơn
    blur = cv2.medianBlur(img, 7)
    
    # Bước 2: Tìm đồng tử (Pupil)
    # Thay vì dùng ngưỡng cố định, ta lấy vùng tối nhất của ảnh làm gợi ý
    min_val, _, min_loc, _ = cv2.minMaxLoc(blur)
    
    # Tạo ngưỡng thích nghi dựa trên giá trị tối nhất để tách đồng tử
    # Đồng tử thường rất tối (gần 0), ta lấy biên trên là min + 30
    thresh_val = min_val + 30 
    _, thresh = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY_INV)
    
    # Tìm vòng tròn đồng tử trong vùng đã tách ngưỡng nhị phân
    pupils = cv2.HoughCircles(
        thresh, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=200,
        param1=50, 
        param2=10, # Ngưỡng tích lũy thấp vì ảnh đã sạch nhiễu sau threshold
        minRadius=20, 
        maxRadius=100
    )
    
    if pupils is None:
        # Nếu Hough thất bại, dùng vị trí tối nhất làm tâm mặc định
        pupil = (float(min_loc[0]), float(min_loc[1]), 50.0)
    else:
        pupil = pupils[0][0]
        
    # Bước 3: Tìm mống mắt (Iris)
    # Mống mắt và đồng tử về mặt sinh học gần như đồng tâm (Concentric)
    px, py, pr = pupil
    
    # Sử dụng Canny để tìm biên, nhưng tăng ngưỡng một chút để lọc nhiễu da mặt
    edges = cv2.Canny(blur, 30, 60)
    cv2.circle(edges, (int(px), int(py)), int(pr + 10), 0, -1)
    
    irises = cv2.HoughCircles(
        edges, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, # Giảm dp để tăng độ chính xác của tâm
        minDist=1, 
        param1=50, 
        param2=20, # Tăng param2 để tránh bắt các đường cong nhiễu ở vùng da
        minRadius=int(pr * 1.5), 
        maxRadius=int(pr * 3.8) # Giới hạn bán kính thực tế hơn (~3.8x đồng tử)
    )
    
    if irises is None:
        # Fallback: Ước lượng mống mắt dựa trên kích thước đồng tử
        iris = (px, py, pr * 2.8)
    else:
        # Lấy danh sách vòng tròn từ Hough (đã được sắp xếp theo độ tin cậy)
        irises = irises[0]
        
        # Tìm vòng tròn đầu tiên (tin cậy nhất) mà có tâm gần đồng tử
        best_iris = None
        for cand in irises:
            dist = np.sqrt((cand[0]-px)**2 + (cand[1]-py)**2)
            if dist < 30: # Chỉ chấp nhận nếu tâm gần đồng tử
                best_iris = cand
                break
        
        if best_iris is None:
            # Nếu không cái nào thỏa mãn, dùng cái đầu tiên và cưỡng chế tâm
            best_iris = irises[0]
            iris = (px, py, best_iris[2])
        else:
            # Cưỡng chế về cùng tâm để Rubber Sheet không bị méo
            iris = (px, py, best_iris[2])
            
    return tuple(map(float, pupil)), tuple(map(float, iris))

def detect_eyelines(img, pupil, iris):
    """
    Tạo mặt nạ (Mask) để loại bỏ vùng nhiễu từ mí mắt và lông mi.
    Đồng thời trả về tọa độ Y của mí mắt để trực quan hóa (đường kẻ màu vàng).
    
    Trả về: (mask, eyeline_y_upper, eyeline_y_lower)
    """
    mask = np.ones_like(img, dtype=np.uint8)
    
    # Tìm các cạnh biên trong ảnh
    edges = cv2.Canny(img, 100, 200)
    
    xp, yp, rp = map(int, pupil)
    xi, yi, ri = map(int, iris)
    
    # Mặc định mí mắt ở ngoài biên ảnh
    eyeline_y_upper = 0
    eyeline_y_lower = img.shape[0]
    
    # Tạo vùng chứa mống mắt (vòng tròn màu trắng trên nền đen)
    iris_mask = np.zeros_like(img, dtype=np.uint8)
    cv2.circle(iris_mask, (xi, yi), ri, 255, -1)
    
    # 1. Xử lý mí mắt trên (Upper Eyelid)
    # Cắt vùng phía trên đồng tử để tìm đường mí mắt
    upper_roi = edges[max(0, yi-ri):yp, max(0, xi-ri):min(img.shape[1], xi+ri)]
    if upper_roi.size > 0:
        # Dòng nào có nhiều pixel cạnh nhất thường là đường mí mắt
        row_sums = np.sum(upper_roi, axis=1)
        eyeline_y_upper = np.argmax(row_sums) + max(0, yi-ri)
        mask[0:eyeline_y_upper, :] = 0
        
    # 2. Xử lý mí mắt dưới (Lower Eyelid)
    lower_roi = edges[yp:min(img.shape[0], yi+ri), max(0, xi-ri):min(img.shape[1], xi+ri)]
    if lower_roi.size > 0:
        row_sums = np.sum(lower_roi, axis=1)
        eyeline_y_lower = np.argmax(row_sums) + yp
        mask[eyeline_y_lower:, :] = 0
        
    # Kết hợp mí mắt và vòng tròn mống mắt để có Mask cuối cùng
    return cv2.bitwise_and(mask, iris_mask), eyeline_y_upper, eyeline_y_lower
