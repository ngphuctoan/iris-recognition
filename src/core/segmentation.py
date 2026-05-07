import cv2
import numpy as np

def segment_iris(img):
    """
    Tach dong tu va mong mat su dung Hough Circles.
    
    Tham so:
    - img: Anh grayscale dau vao.
    
    Tra ve:
    - tuple: (pupil_circle, iris_circle) trong do moi circle la (x, y, r).
    """
    if img is None:
        raise TypeError("Anh dau vao khong duoc la None")
        
    # 1. Lam mo anh de giam nhiễu biên
    blur = cv2.medianBlur(img, 7)
    
    # 2. Tim dong tu (vung toi nhat)
    # Su dung threshold de highlight vung toi
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
    
    # Tim dong tu trong vung thresh
    pupils = cv2.HoughCircles(
        thresh, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=200,
        param1=50, 
        param2=10, # Threshold nho hon vi thresh da loai bo nhieu nhiễu
        minRadius=20, 
        maxRadius=100
    )
    
    if pupils is None:
        # Fallback neu thresh failed
        pupils = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 200, param1=50, param2=30, minRadius=20, maxRadius=100)
        
    if pupils is None:
        pupil = (img.shape[1]//2, img.shape[0]//2, 50)
    else:
        pupil = pupils[0][0]
        
    # 3. Tim mong mat (vung bao quanh dong tu)
    # Su dung tam cua dong tu de gioi han vung tim kiem (gan nhu dong tam)
    px, py, pr = pupil
    irises = cv2.HoughCircles(
        blur, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=200,
        param1=50, 
        param2=30, 
        minRadius=int(pr * 2), # Mong mat it nhat gap doi dong tu
        maxRadius=300
    )
    
    if irises is None:
        iris = (px, py, pr * 3)
    else:
        # Chon hinh tron co tam gan dong tu nhat trong so cac ket qua
        best_iris = irises[0][0]
        min_dist = np.sqrt((best_iris[0] - px)**2 + (best_iris[1] - py)**2)
        
        for cand in irises[0]:
            dist = np.sqrt((cand[0] - px)**2 + (cand[1] - py)**2)
            # Neu tam qua xa dong tu (> 50px), co the la nham
            if dist < min_dist and dist < 50:
                min_dist = dist
                best_iris = cand
        
        # Neu best_iris van qua xa, dung gia tri mac dinh nhung giu tam dong tu
        dist = np.sqrt((best_iris[0] - px)**2 + (best_iris[1] - py)**2)
        if dist > 50:
            iris = (px, py, best_iris[2])
        else:
            iris = best_iris
        
    return tuple(map(float, pupil)), tuple(map(float, iris))

def detect_eyelines(img, pupil, iris):
    """
    Phat hien mi mat va long mi de tao mat na mask.
    Su dung Canny + Horizontal lines (KISS).
    """
    mask = np.ones_like(img, dtype=np.uint8)
    
    # 1. Su dung Canny de tim cac canh (long mi, mi mat)
    edges = cv2.Canny(img, 100, 200)
    
    # 2. Tap trung vao vung ben tren va ben duoi dong tu
    # Mi mat thuong xuat hien o phia tren va phia duoi
    xp, yp, rp = map(int, pupil)
    xi, yi, ri = map(int, iris)
    
    # Tao vung quan tam (ROI) la hinh tron mong mat
    iris_mask = np.zeros_like(img, dtype=np.uint8)
    cv2.circle(iris_mask, (xi, yi), ri, 255, -1)
    
    # 3. Tim mi mat tren (upper eyelid)
    # Don gian hoa: Tim cac canh nam ngang phia tren dong tu
    upper_roi = edges[max(0, yi-ri):yp, max(0, xi-ri):min(img.shape[1], xi+ri)]
    if upper_roi.size > 0:
        # Lay dong co nhieu canh nhat lam duong bien mi mat
        row_sums = np.sum(upper_roi, axis=1)
        eyeline_y = np.argmax(row_sums) + (yi-ri)
        mask[0:eyeline_y, :] = 0
        
    # 4. Tim mi mat duoi (lower eyelid)
    lower_roi = edges[yp:min(img.shape[0], yi+ri), max(0, xi-ri):min(img.shape[1], xi+ri)]
    if lower_roi.size > 0:
        row_sums = np.sum(lower_roi, axis=1)
        eyeline_y = np.argmax(row_sums) + yp
        mask[eyeline_y:, :] = 0
        
    # Ket hop voi vung hinh tron mong mat
    final_mask = cv2.bitwise_and(mask, iris_mask)
    
    return final_mask
