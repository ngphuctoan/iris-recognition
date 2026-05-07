import os
import numpy as np
from pathlib import Path
from src.core.io import load_image
from src.core.segmentation import segment_iris, detect_eyelines
from src.core.normalization import normalize_iris
from src.core.encoding import encode_iris
from src.core.matching import match_codes
from tqdm import tqdm

def run_benchmark():
    subset_dir = Path("data/subset")
    subject_dirs = sorted([d for d in subset_dir.iterdir() if d.is_dir()])
    
    print(f"Bat dau Benchmark tren {len(subject_dirs)} subjects...")
    
    templates = {}
    
    # 1. Trich xuat template cho tat ca anh
    print("Dang trich xuat templates...")
    for subject in tqdm(subject_dirs):
        imgs = sorted(list(subject.glob("*.jpg")))
        templates[subject.name] = []
        for img_path in imgs:
            try:
                img = load_image(str(img_path))
                p, i = segment_iris(img)
                mask_raw = detect_eyelines(img, p, i)
                norm, norm_mask = normalize_iris(img, p, i, mask=mask_raw)
                code, mask = encode_iris(norm, mask=norm_mask)
                templates[subject.name].append((code, mask))
            except Exception as e:
                # print(f"Loi tai {img_path}: {e}")
                pass

    genuine_scores = []
    imposter_scores = []
    
    # 2. Tinh toan Genuine Scores (Giong nhau)
    print("Dang tinh Genuine Scores...")
    for s_name, s_temps in templates.items():
        for i in range(len(s_temps)):
            for j in range(i + 1, len(s_temps)):
                _, dist = match_codes(s_temps[i][0], s_temps[j][0], s_temps[i][1], s_temps[j][1])
                genuine_scores.append(dist)
                
    # 3. Tinh toan Imposter Scores (Khac nhau)
    print("Dang tinh Imposter Scores...")
    s_names = list(templates.keys())
    for i in range(len(s_names)):
        for j in range(i + 1, len(s_names)):
            # So khop anh dau tien cua subject i voi anh dau tien cua subject j
            if templates[s_names[i]] and templates[s_names[j]]:
                t1 = templates[s_names[i]][0]
                t2 = templates[s_names[j]][0]
                _, dist = match_codes(t1[0], t2[0], t1[1], t2[1])
                imposter_scores.append(dist)

    # 4. Phan tich ket qua tai nguong threshold = 0.35
    threshold = 0.35
    false_rejections = sum(1 for s in genuine_scores if s >= threshold)
    false_acceptances = sum(1 for s in imposter_scores if s < threshold)
    
    frr = false_rejections / len(genuine_scores) if genuine_scores else 0
    far = false_acceptances / len(imposter_scores) if imposter_scores else 0
    
    print("\n--- KET QUA BENCHMARK ---")
    print(f"Tong so Genuine tests: {len(genuine_scores)}")
    print(f"Tong so Imposter tests: {len(imposter_scores)}")
    print(f"Threshold: {threshold}")
    print(f"False Rejections: {false_rejections} (FRR: {frr*100:.2f}%)")
    print(f"False Acceptances: {false_acceptances} (FAR: {far*100:.2f}%)")
    print(f"Average Genuine Distance: {np.mean(genuine_scores):.4f}")
    print(f"Average Imposter Distance: {np.mean(imposter_scores):.4f}")

if __name__ == "__main__":
    run_benchmark()
