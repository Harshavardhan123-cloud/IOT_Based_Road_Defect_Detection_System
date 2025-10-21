import os
import cv2
import numpy as np
from functools import partial
from multiprocessing import Pool, cpu_count

# ----- USER SETTINGS -----
ROOT_DIR = "split"  # dataset root
SUBFOLDERS = ["train", "test", "val"]
OUTPUT_SUFFIX = "_fast2"             # e.g. "train_fast"
NUM_PROCESSES = max(1, cpu_count() - 1)
# --------------------------

# Adjusted filter params for lighter smoothing
BILAT_D = 5
BILAT_SIG_COLOR = 50
BILAT_SIG_SPACE = 50
CLAHE_CLIP = 1.5
CLAHE_GRID = (8, 8)
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
INPAINT_RADIUS = 1


def preprocess_image(path, out_root):
    """Load, process, and save one image with reduced filter intensity."""
    try:
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return False
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1) Shadow suppression via smaller closing on L channel
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        closed = cv2.morphologyEx(l, cv2.MORPH_CLOSE, MORPH_KERNEL)
        lab = cv2.merge((closed, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # 2) Lighter bilateral filter
        img = cv2.bilateralFilter(img, BILAT_D, BILAT_SIG_COLOR, BILAT_SIG_SPACE)

        # 3) CLAHE with reduced clip limit
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # 4) Minor morphological opening/closing and light inpainting
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, MORPH_KERNEL)
        img = cv2.inpaint(img, cv2.bitwise_not(morph), INPAINT_RADIUS, cv2.INPAINT_TELEA)

        # Save
        rel_path = os.path.relpath(path, start=out_root.replace(OUTPUT_SUFFIX, ""))
        save_dir = os.path.join(out_root, os.path.dirname(rel_path))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(out_root, rel_path)
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return True
    except Exception:
        return False


def gather_images(root, subfolder):
    img_list = []
    base = os.path.join(root, subfolder)
    for dirpath, _, files in os.walk(base):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                img_list.append(os.path.join(dirpath, f))
    return img_list


if __name__ == "__main__":
    for sub in SUBFOLDERS:
        imgs = gather_images(ROOT_DIR, sub)
        out_root = os.path.join(ROOT_DIR, sub + OUTPUT_SUFFIX)
        print(f"[{sub}] Processing {len(imgs)} images → {out_root}")

        worker = partial(preprocess_image, out_root=out_root)
        with Pool(processes=NUM_PROCESSES) as p:
            for idx, success in enumerate(p.imap_unordered(worker, imgs), 1):
                if idx % 100 == 0 or not success:
                    status = "OK" if success else "FAIL"
                    print(f"  [{sub}] {idx}/{len(imgs)} → {status}")

    print("Completed with reduced filter intensity.")