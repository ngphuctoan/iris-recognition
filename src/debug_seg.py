from src.core.io import load_image
from src.core.segmentation import segment_iris
from src.utils.visualization import show_segmentation
from pathlib import Path

def debug_segmentation():
    subset_dir = Path("data/subset")
    # Lay 5 anh ngau nhien
    img_paths = list(subset_dir.glob("**/*.jpg"))[:5]
    
    for path in img_paths:
        img = load_image(str(path))
        p, i = segment_iris(img)
        print(f"File: {path.name} | Pupil: {p} | Iris: {i}")
        # show_segmentation(img, p, i, title=f"Debug: {path.name}")

if __name__ == "__main__":
    debug_segmentation()
