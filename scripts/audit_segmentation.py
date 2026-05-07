import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from src.core.io import load_image
from src.core.segmentation import segment_iris

def audit_segmentation(num_samples=10, output_path="temp/segmentation_audit.png"):
    """
    Chon ngau nhien cac mau tu dataset va kiem tra ket qua tach vung (Segmentation).
    Luu ket qua vao mot file anh duy nhat de de dang kiem tra.
    """
    # Dam bao thu muc temp ton tai
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    samples = []
    attempts = 0
    while len(samples) < num_samples and attempts < 100:
        # Gioi han trong khoang 0-49 vi data/subset chi co 50 subjects
        sub_id = f"{random.randint(0, 49):03d}"
        seq = f"{random.randint(0, 3):02d}" # Subset thuong co it sequence hon
        side = random.choice(["L", "R"])
        
        try:
            # load_image mac dinh se dung data/subset
            img = load_image(subject_id=sub_id, side=side, sequence=seq)
            samples.append((img, sub_id, side, seq))
        except Exception:
            pass
        attempts += 1
        
    if not samples:
        print("Error: Khong the load mau tu dataset. Kiem tra ket noi mang hoac cau truc dataset.")
        return

    # Tao grid de hien thi
    cols = 5
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    fig.patch.set_facecolor('#fcfcfc') # Bone White
    axes = axes.flatten()

    for i, (img, sub_id, side, seq) in enumerate(samples):
        # Chay segmentation
        try:
            p, iris_c = segment_iris(img)
            
            # Ve ket qua
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            px, py, pr = map(int, p)
            ix, iy, ir = map(int, iris_c)
            
            # Royal Blue cho Iris, Teal cho Pupil
            cv2.circle(vis, (ix, iy), ir, (133, 64, 0), 3) 
            cv2.circle(vis, (px, py), pr, (102, 98, 0), 2)
            
            axes[i].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"S{sub_id}_{side}_{seq}", fontsize=10, fontweight='bold', color='#2d3436')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"FAIL: {str(e)}", ha='center', va='center')
            
        axes[i].axis('off')

    # An cac subplot thua
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Audit complete. Results saved to: {output_path}")

if __name__ == "__main__":
    audit_segmentation()
