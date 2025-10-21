import os
import cv2
import numpy as np
from functools import partial
from multiprocessing import Pool, cpu_count

# ----- USER SETTINGS -----
ROOT_DIR = "split"  # dataset root
SUBFOLDERS = ["train", "test", "val"]
OUTPUT_SUFFIX = "_fast5"             # e.g. "train_fast"
NUM_PROCESSES = max(1, cpu_count() - 1)
# --------------------------

# Parameters for custom smoothing
CUSTOM_KERNEL = 5
GRADIENT_THRESHOLD = 20
CLAHE_CLIP = 1.5
CLAHE_GRID = (8, 8)
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
INPAINT_RADIUS = 1


def get_mean_kernel(k):
    """
    Create and print a custom kernel where center is 0.01 and rest are 0.4 normalized.
    """
    kernel = np.full((k, k), 0.4, dtype=np.float32)
    center = k // 2
    kernel[center, center] = 0.01
    kernel /= np.sum(kernel)
    print(f"Custom mean filter kernel (size {k}x{k}):")
    print(kernel)
    return kernel


def custom_smooth(img, kernel_size=CUSTOM_KERNEL, grad_thresh=GRADIENT_THRESHOLD):
    """
    Custom edge-aware blur:
    - if local gradient > threshold, use median filter
    - else use custom mean filter
    """
    pad = kernel_size // 2
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    h, w = img.shape[:2]
    out = np.zeros_like(img)

    mean_kernel = get_mean_kernel(kernel_size)

    for y in range(h):
        for x in range(w):
            patch = img_padded[y:y+kernel_size, x:x+kernel_size]
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            gx = cv2.Sobel(gray_patch, cv2.CV_64F, 1, 0, ksize=3)[pad, pad]
            gy = cv2.Sobel(gray_patch, cv2.CV_64F, 0, 1, ksize=3)[pad, pad]
            grad_mag = np.sqrt(gx * gx + gy * gy)
            pixels = patch.reshape(-1, 3)
            if grad_mag > grad_thresh:
                val = np.median(pixels, axis=0)
            else:
                val = np.sum(pixels * mean_kernel.flatten()[:, None], axis=0)
            out[y, x] = val

    return out.astype(np.uint8)


def preprocess_image(path, out_root):
    """Load, apply custom smooth + CLAHE + light morph, save RGB image."""
    try:
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return False
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1) Custom smoothing based on local gradients
        img = custom_smooth(img)

        # 2) CLAHE on L channel for contrast boost
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # 3) Minor morphological cleanup and light inpainting
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, MORPH_KERNEL)
        img = cv2.inpaint(img, cv2.bitwise_not(morph), INPAINT_RADIUS, cv2.INPAINT_TELEA)

        # Save result
        rel = os.path.relpath(path, start=out_root.replace(OUTPUT_SUFFIX, ""))
        save_dir = os.path.join(out_root, os.path.dirname(rel))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(out_root, rel)
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return True
    except Exception:
        return False


def gather_images(root, subfolder):
    lst = []
    base = os.path.join(root, subfolder)
    for dp, _, files in os.walk(base):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                lst.append(os.path.join(dp, f))
    return lst


if __name__ == "__main__":
    get_mean_kernel(CUSTOM_KERNEL)
    print("Note: median filter is non-linear and has no fixed kernel representation.")

    for sub in SUBFOLDERS:
        imgs = gather_images(ROOT_DIR, sub)
        out_root = os.path.join(ROOT_DIR, sub + OUTPUT_SUFFIX)
        print(f"[{sub}] Processing {len(imgs)} images → {out_root}")

        worker = partial(preprocess_image, out_root=out_root)
        with Pool(processes=NUM_PROCESSES) as p:
            for idx, ok in enumerate(p.imap_unordered(worker, imgs), 1):
                if idx % 100 == 0 or not ok:
                    print(f"  [{sub}] {idx}/{len(imgs)} → {'OK' if ok else 'FAIL'}")

    print("Custom smoothing pipeline complete.")
