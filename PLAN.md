# Iris Recognition System Implementation Plan (V3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> [!IMPORTANT]
> **Key Project Requirements:**
> - **Algorithm**: Phải tuân thủ chặt chẽ thuật toán gốc của Daugman.
> - **Segmentation**: Sử dụng kỹ thuật Computer Vision (Hough circles, thresholding, v.v.).
> - **Accuracy**: Đạt chỉ số FRR và FAR hợp lý (kiểm chứng qua TDD).
> - **KISS Principle**: Giữ thuật toán đơn giản, dễ hiểu, tránh phức tạp hóa vấn đề.
> - **Documentation**: Toàn bộ mã nguồn phải được comment bằng tiếng Việt (theo phong cách KISS).
> - **Methodology**: Áp dụng Test-Driven Development (TDD) xuyên suốt quá trình.
> - **Skills**: Tận dụng tối đa các công cụ và kỹ năng hỗ trợ trong `.agents`.

**Goal:** Build a robust iris recognition system using Daugman's algorithm with a FastAPI web interface and SQLModel storage, achieving high accuracy on the CASIA-Iris-Thousand dataset.

**Architecture:** Modular pipeline consisting of Computer Vision segmentation (Hough Circles + Eyeline detection), Rubber Sheet normalization, Gabor Filter encoding, and Hamming Distance matching.

**Tech Stack:** OpenCV, NumPy, FastAPI, SQLModel, Matplotlib, Pytest.

---

## 0/ Dataset Overview (CASIA-Iris-Thousand)
Cấu trúc thư mục của bộ dữ liệu CASIA-Iris-Thousand:
```text
CASIA-Iris-Thousand/
├── 000/
│   ├── L/
│   │   ├── 000_L_1.jpg
│   │   └── ...
│   └── R/
│       ├── 000_R_1.jpg
│       └── ...
├── 001/
│   ├── L/ ...
│   └── R/ ...
└── iris_thousands.csv (Metadata mapping - nếu có)
```

## 1/ Project Overview
Hệ thống nhận diện mống mắt (Iris Recognition System) được xây dựng dựa trên thuật toán kinh điển của Daugman. Hệ thống thực hiện các bước: Tách vùng mống mắt (Segmentation), Chuẩn hóa (Normalization), Mã hóa (Encoding) và So khớp (Matching). Việc viết comment tiếng Việt và tối ưu code theo KISS sẽ được thực hiện song song với quá trình code.

## 2/ Skills to be Used (from .agents)
- **python-anti-patterns**: Tránh các lỗi lập trình Python phổ biến để code chạy ổn định.
- **python-design-patterns**: Áp dụng KISS, SoC, và SRP để hệ thống dễ bảo trì.
- **python-error-handling**: Xử lý ngoại lệ chặt chẽ khi xử lý ảnh lỗi hoặc DB.
- **python-type-safety**: Sử dụng type hints để code minh bạch và dễ debug.
- **test-driven-development**: Viết test trước khi implement core logic.
- **python-testing-patterns**: Cấu trúc bộ test chuyên nghiệp với pytest.
- **uv**: Quản lý project và dependencies nhanh chóng.

## 3/ Project Structure
```text
iris-recognition/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── io.py            # load_image()
│   │   ├── segmentation.py  # segment_iris(), find_pupil(), find_iris()
│   │   ├── normalization.py # normalize_iris()
│   │   ├── encoding.py      # encode_iris()
│   │   └── matching.py      # match_codes(), hamming_distance()
│   ├── api/
│   │   ├── main.py          # FastAPI routes
│   │   └── models.py        # SQLModel schemas
│   ├── utils/
│   │   └── visualization.py # show_results(), plot_debug_match()
│   └── main.py              # CLI entry point cho testing
├── tests/                   # Unit tests và Accuracy tests (Benchmark)
├── data/                    # Dataset subset (gitignored)
├── pyproject.toml
└── PLAN.md
```

## 4/ The Recognition Algorithm & Implementation Logic

Hệ thống triển khai thuật toán Daugman với logic chi tiết như sau:

1.  **Đọc ảnh (`load_image`)**: Chuyển ảnh sang Grayscale và kiểm tra tính hợp lệ.
2.  **Tách vùng mống mắt (`segment_iris`)**: 
    - Sử dụng `cv2.HoughCircles` để tìm đồng tử và mống mắt. 
    - **Eyeline Detector**: Phát hiện mí mắt bằng Canny + Horizontal lines để tạo Mask.
    - **Cropping**: Cắt ảnh xung quanh vùng mống mắt để tối ưu tính toán.
3.  **Chuẩn hóa (`normalize_iris`)**: 
    - Áp dụng Daugman's Rubber Sheet Model để chuyển vùng mống mắt sang hình chữ nhật (64x512).
4.  **Mã hóa (`encode_iris`)**: 
    - Trích xuất đặc trưng bằng 2D Gabor Filters và binarize pha của tín hiệu.
5.  **So khớp (`match_codes`)**: 
    - Tính `Hamming Distance` kết hợp với Mask để loại bỏ vùng nhiễu.

### Implementation Logic (Python Draft)

```python
import cv2
import numpy as np

def load_image(image_path: str):
    """Doc anh grayscale tu duong dan."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Khong tim thay anh tai {image_path}")
    return img

def segment_iris(img):
    """Tach dong tu va mong mat bang Hough Circles."""
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    # Tim dong tu (pupil)
    pupil = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 200, param1=50, param2=30, minRadius=20, maxRadius=80)
    # Tim mong mat (iris)
    iris = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 200, param1=50, param2=30, minRadius=100, maxRadius=250)
    return pupil[0][0], iris[0][0]

def normalize_iris(img, pupil, iris):
    """Unwrap vung mong mat sang hinh chu nhat 64x512."""
    res_r, res_theta = 64, 512
    norm_img = np.zeros((res_r, res_theta), dtype=np.uint8)
    
    xp, yp, rp = pupil
    xi, yi, ri = iris
    
    for r in range(res_r):
        # r_norm chay tu 0 den 1 (ti le ban kinh)
        r_norm = r / res_r
        for t in range(res_theta):
            theta = 2 * np.pi * t / res_theta
            # Toa do tren bien dong tu va mống mắt
            x_p_edge = xp + rp * np.cos(theta)
            y_p_edge = yp + rp * np.sin(theta)
            x_i_edge = xi + ri * np.cos(theta)
            y_i_edge = yi + ri * np.sin(theta)
            
            # Mapping Rubber Sheet
            x = int((1 - r_norm) * x_p_edge + r_norm * x_i_edge)
            y = int((1 - r_norm) * y_p_edge + r_norm * y_i_edge)
            
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                norm_img[r, t] = img[y, x]
    return norm_img

def match_codes(code1, code2, mask1, mask2):
    """Tinh Hamming Distance giua 2 IrisCode."""
    # XOR de tim cac bit khac nhau
    diff = cv2.bitwise_xor(code1, code2)
    # Chi xet cac vung hop le (mask)
    valid = cv2.bitwise_and(mask1, mask2)
    diff_in_valid = cv2.bitwise_and(diff, valid)
    
    score = np.sum(diff_in_valid) / np.sum(valid)
    return score
```

## 5/ SQL Model Schema (ASCII)
```text
+-------------------+       +-----------------------+
|       User        |       |      IrisTemplate     |
+-------------------+       +-----------------------+
| id (PK)           |<-----+| id (PK)               |
| name (String)     |       | user_id (FK)          |
| identity (String) |       | eye_side (String)     |
+-------------------+       | iris_code (HexStr)    |
                            | mask (HexStr)         |
                            +-----------------------+
```

## 6/ Web Interface
- **Công nghệ**: FastAPI (Backend), Jinja2/HTML/CSS (Frontend).
- **Tính năng**: Enrollment, Identification, và Visualization (Debug View).

## 7/ Phases

### Phase 0: Create Testing
- [ ] Thiết lập `pytest` và cấu trúc `tests/`.
- [ ] Chuẩn bị Benchmark subset (200+ ảnh).
- [ ] Viết test cho `load_image` (lỗi file, định dạng không hỗ trợ).

### Phase A: Building the Core Algorithm
- [ ] Task 1: Khởi tạo project structure.
- [ ] Task 2: Implement `io.py` (`load_image`) & `segmentation.py` (`segment_iris`).
- [ ] Task 3: Implement `normalization.py` (Rubber Sheet logic).
- [ ] Task 4: Implement `encoding.py` & `matching.py`. **Thêm Debug Plotting** (hiển thị XOR result và IrisCode visualization).

### Phase B: Improving the Accuracy
- [ ] Task 5: Implement **Eyeline Detector** và **Image Cropping** (Canny + Horizontal Lines).
- [ ] Task 6: Tối ưu hóa tham số Gabor.
- [ ] Task 7: Chạy Benchmark diện rộng.

### Phase C: Building the Web Interface
- [ ] Task 8: Database & SQLModel setup.
- [ ] Task 9: FastAPI endpoints & Web UI (Premium aesthetics).

### Phase D: Polish
- [ ] Task 10: Rà soát comment tiếng Việt & KISS.

## 8/ Notes + Decisions
- **Eyeline Detection**: Sử dụng Canny + Horizontal Line cho đến khi gặp điểm nghẽn (bottleneck).
