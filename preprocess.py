import os
import cv2
import shutil
import glob
import numpy as np
import yaml

def apply_preprocessing(image_path, sigma=3):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None
    
    # Convert image to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Apply Gaussian Blur in HSV space
    blurred_hsv = cv2.GaussianBlur(img_hsv, (0, 0), sigmaX=sigma)
    
    # Apply Histogram Equalization to the V (Value) channel
    blurred_hsv[:, :, 2] = cv2.equalizeHist(blurred_hsv[:, :, 2])  # Apply equalization to the Value channel
    
    # Convert back to BGR for saving
    img_eq = cv2.cvtColor(blurred_hsv, cv2.COLOR_HSV2BGR)
    
    return img_eq

def process_images_and_labels(image_dir, label_dir, out_image_dir, out_label_dir, sigma=3):
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    for img_path in glob.glob(os.path.join(image_dir, '*')):
        fname = os.path.basename(img_path)
        name, _ = os.path.splitext(fname)

        out_img_path = os.path.join(out_image_dir, fname)
        out_lbl_path = os.path.join(out_label_dir, f"{name}.txt")
        lbl_path = os.path.join(label_dir, f"{name}.txt")

        # Process image
        processed_img = apply_preprocessing(img_path, sigma)
        if processed_img is not None:
            cv2.imwrite(out_img_path, processed_img)
            print(f"Saved image: {out_img_path}")

        # Copy label if the file paths are different
        if os.path.exists(lbl_path) and lbl_path != out_lbl_path:
            shutil.copy(lbl_path, out_lbl_path)
            print(f"Copied label: {out_lbl_path}")
        else:
            print(f"Warning: No label found or label is already in the destination.")

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data, path):
    with open(path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

def main():
    original_yaml = "Fin_Aug_Dataset/data.yaml"
    dataset = load_yaml(original_yaml)
    class_names = dataset.get("names", [])

    # Prepare output base
    out_base = "processed"
    os.makedirs(out_base, exist_ok=True)

    updated_yaml = {
        "train": "",
        "valid": "",
        "test": "",
        "nc": len(class_names),
        "names": class_names
    }

    for split in ["train", "valid", "test"]:
        if split not in dataset:
            continue

        image_dir = dataset[split]
        label_dir = image_dir.replace("images", "labels")  # assume same pattern

        out_image_dir = os.path.join(out_base, image_dir)
        out_label_dir = os.path.join(out_base, label_dir)

        # Process this split
        print(f"\nProcessing {split}:")
        process_images_and_labels(image_dir, label_dir, out_image_dir, out_label_dir)

        # Update path in new yaml
        updated_yaml[split] = out_image_dir

    # Save new data.yaml
    new_yaml_path = os.path.join(out_base, "data.yaml")
    save_yaml(updated_yaml, new_yaml_path)
    print(f"\n✅ New data.yaml saved at: {new_yaml_path}")

if __name__ == "__main__":
    main()
