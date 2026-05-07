import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_segmentation(img, pupil, iris, title="Segmentation Result"):
    """Hien thi ket qua tach mong mat."""
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Ve dong tu (do)
    cv2.circle(out, (int(pupil[0]), int(pupil[1])), int(pupil[2]), (0, 0, 255), 2)
    # Ve mong mat (xanh)
    cv2.circle(out, (int(iris[0]), int(iris[1])), int(iris[2]), (0, 255, 0), 2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_normalization(norm_img, title="Normalized Iris"):
    """Hien thi mống mắt đã trải phẳng."""
    plt.figure(figsize=(15, 5))
    plt.imshow(norm_img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_debug_match(code1, code2, mask1, mask2, distance, title="Match Debug"):
    """Hien thi ket qua so khop va tim vung tot nhat (shift)."""
    min_dist = 1.0
    best_shift = 0
    
    # Tim shift tot nhat de hien thi
    for pixel_shift in range(-30, 31):
        shift = pixel_shift * 2
        s_code = np.roll(code2, shift, axis=1)
        s_mask = np.roll(mask2, shift, axis=1)
        # Tinh nhanh dist
        xor_res = cv2.bitwise_xor(code1, s_code)
        valid = cv2.bitwise_and(mask1, s_mask)
        dist = np.sum(cv2.bitwise_and(xor_res, valid)) / np.sum(valid)
        if dist < min_dist:
            min_dist = dist
            best_shift = shift
            
    # Hien thi voi shift tot nhat
    s_code = np.roll(code2, best_shift, axis=1)
    s_mask = np.roll(mask2, best_shift, axis=1)
    diff = cv2.bitwise_xor(code1, s_code)
    valid = cv2.bitwise_and(mask1, s_mask)
    diff_in_valid = cv2.bitwise_and(diff, valid)
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    axes[0].imshow(code1, cmap='gray'); axes[0].set_title("IrisCode 1"); axes[0].axis('off')
    axes[1].imshow(s_code, cmap='gray'); axes[1].set_title(f"IrisCode 2 (Shifted {best_shift})"); axes[1].axis('off')
    axes[2].imshow(diff_in_valid, cmap='hot'); axes[2].set_title(f"Differences (Min Distance: {min_dist:.4f})")
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
