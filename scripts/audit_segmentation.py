import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from src.core.io import load_image
from src.core.segmentation import segment_iris, detect_eyelines

def audit_segmentation(num_samples=10, output_path="temp/segmentation_audit.png"):
    """
    Chọn ngẫu nhiên các mẫu từ dataset và kiểm tra kết quả tách vùng (Segmentation).
    Vẽ thêm các đường mí mắt màu vàng để kiểm tra Eyeline Detector.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    samples = []
    attempts = 0
    while len(samples) < num_samples and attempts < 100:
        sub_id = f"{random.randint(0, 49):03d}"
        seq = f"{random.randint(0, 3):02d}"
        side = random.choice(["L", "R"])
        
        try:
            img = load_image(subject_id=sub_id, side=side, sequence=seq)
            samples.append((img, sub_id, side, seq))
        except Exception:
            pass
        attempts += 1
        
    if not samples:
        print("Error: Khong the load mau tu dataset.")
        return

    cols = 5
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    fig.patch.set_facecolor('#fcfcfc')
    axes = axes.flatten()

    for i, (img, sub_id, side, seq) in enumerate(samples):
        try:
            p, iris_c = segment_iris(img)
            # Lấy thêm tọa độ mí mắt
            _, y_up, y_down = detect_eyelines(img, p, iris_c)
            
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            px, py, pr = map(int, p)
            ix, iy, ir = map(int, iris_c)
            
            # Vẽ đồng tử (Teal) và mống mắt (Royal Blue)
            cv2.circle(vis, (ix, iy), ir, (133, 64, 0), 3) 
            cv2.circle(vis, (px, py), pr, (102, 98, 0), 2)
            
            # Vẽ đường mí mắt (Vàng)
            cv2.line(vis, (0, int(y_up)), (img.shape[1], int(y_up)), (0, 255, 255), 2)
            cv2.line(vis, (0, int(y_down)), (img.shape[1], int(y_down)), (0, 255, 255), 2)
            
            axes[i].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"S{sub_id}_{side}_{seq}", fontsize=10, fontweight='bold')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"FAIL: {str(e)}", ha='center', va='center')
            
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Audit complete with Yellow Eyelines. Results saved to: {output_path}")

if __name__ == "__main__":
    audit_segmentation()
