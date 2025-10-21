# %%
import os
import numpy as np
import cv2
import math
from collections import defaultdict
import random
import matplotlib.pyplot as plt

# %%
# Directories
main_dir = "Fin_Dataset/valid"
images_dir = os.path.join(main_dir, "images")
labels_dir = os.path.join(main_dir, "labels")

# %% [markdown]
# ### Remove NULL images and labels

# %%
for label_file in os.listdir(labels_dir):
    label_path = os.path.join(labels_dir, label_file)
    # Check if the label file is empty
    if os.path.isfile(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        if not lines:
            os.remove(label_path)
            image_file = os.path.splitext(label_file)[0] + ".jpg"  # Change extension if needed
            image_path = os.path.join(images_dir, image_file)
            if os.path.exists(image_path):
                os.remove(image_path)
                    
print("Empty labels and their corresponding images have been removed.")

# %% [markdown]
# ### Count the number of instances of each class

# %%
class_names = ['Cracks', 'Edge_Settling', 'No_Road', 'Potholes', 'Raveling', 'Rutting']
num_classes = len(class_names)

# %%
def count_class_instances(labels_dir, class_names):
    class_counts = np.zeros(len(class_names), dtype=int)
    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)
        if os.path.isfile(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
    return {i: count for i, count in enumerate(class_counts)}

# %%
original_class_counts = count_class_instances(labels_dir, class_names)
print("\nOriginal class distribution:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {original_class_counts[i]}")

# %% [markdown]
# ### Performing Augmentation

# %%
augmented_images_dir = os.path.join(main_dir, "aug_images")
augmented_labels_dir = os.path.join(main_dir, "aug_labels")
os.makedirs(augmented_images_dir, exist_ok=True)
os.makedirs(augmented_labels_dir, exist_ok=True)

# %% [markdown]
# #### Control Factor

# %%
augmentation_factor = 7490 # Adjust this to control the total count
target_instances_per_class = augmentation_factor // num_classes
print("\nTarget instances per class:", target_instances_per_class)

# %%
required_augmentations = {
    class_id: max(target_instances_per_class - original_class_counts[class_id], 0)
    for class_id in range(num_classes)
}

print("\nRequired augmentations per class:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {required_augmentations[i]}")

# %%
# Track how many images exist per class
image_counts_per_class = defaultdict(int)
for label_file in os.listdir(labels_dir):
    label_path = os.path.join(labels_dir, label_file)
    if os.path.isfile(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            image_classes = set(int(line.split()[0]) for line in lines)
            for class_id in image_classes:
                image_counts_per_class[class_id] += 1

# %%
# Compute augmentations per image
augmentations_per_image = {}
for class_id in range(num_classes):
    if image_counts_per_class[class_id] > 0:
        augmentations_per_image[class_id] = math.ceil(required_augmentations[class_id] / image_counts_per_class[class_id])
    else:
        augmentations_per_image[class_id] = 0

print("\nAugmentations per image:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {augmentations_per_image[i]}")

# %% [markdown]
# ### Augmentation Function with Label Transformation
#
# This function applies a series of augmentations:
#  - **Geometric augmentations:** rotation, shift-scale, and horizontal flip.  
#    The corresponding bounding boxes are updated by computing the transformed positions of their corner points.
#  - **Photometric augmentations:** brightness/contrast, hue/saturation adjustments, blur, sharpness, gamma, and CLAHE.
#
# The labels (in YOLO format) are updated only for geometric changes.

# %%
def augment_image_and_labels(image, label_lines):
    h, w = image.shape[:2]
    
    # Parse labels into bounding boxes in absolute coordinates
    # Format: [class, xmin, ymin, xmax, ymax]
    bboxes = []
    for line in label_lines:
        parts = line.strip().split()
        cls = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        xmin = (cx - bw/2) * w
        ymin = (cy - bh/2) * h
        xmax = (cx + bw/2) * w
        ymax = (cy + bh/2) * h
        bboxes.append([cls, xmin, ymin, xmax, ymax])
    
    # Initialize composite transformation matrix as identity (3x3)
    M_total = np.eye(3)

    # --- Geometric Transformations ---
    # Rotation
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)  # 2x3 matrix
        # Convert to 3x3
        M_rot = np.vstack([M, [0, 0, 1]])
        M_total = M_rot @ M_total
        image = cv2.warpAffine(image, M, (w, h))
    
    # Shift & Scale (combined)
    if random.random() < 0.5:
        scale = random.uniform(0.9, 1.1)
        tx = random.randint(-20, 20)
        ty = random.randint(-20, 20)
        M_shift_scale = np.array([[scale, 0, tx],
                                  [0, scale, ty],
                                  [0,    0,  1]])
        M_total = M_shift_scale @ M_total
        image = cv2.warpAffine(image, M_shift_scale[:2, :], (w, h))
    
    # Horizontal Flip
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        M_flip = np.array([[-1, 0, w],
                           [0, 1, 0],
                           [0, 0, 1]])
        M_total = M_flip @ M_total

    # Update bounding boxes using the composite geometric transformation
    new_label_lines = []
    for bbox in bboxes:
        cls, xmin, ymin, xmax, ymax = bbox
        # Create an array of the four corner points
        pts = np.array([[xmin, ymin],
                        [xmax, ymin],
                        [xmax, ymax],
                        [xmin, ymax]], dtype=np.float32)
        # Convert points to homogeneous coordinates (3 x N)
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        pts_hom = np.hstack([pts, ones]).T  # shape: (3,4)
        pts_trans = (M_total @ pts_hom).T  # transformed points, shape: (4,3)
        pts_trans = pts_trans[:, :2]  # use only (x,y)

        x_min_new = np.min(pts_trans[:,0])
        y_min_new = np.min(pts_trans[:,1])
        x_max_new = np.max(pts_trans[:,0])
        y_max_new = np.max(pts_trans[:,1])
        
        # Clip coordinates to lie within the image dimensions
        x_min_new = max(0, min(w, x_min_new))
        y_min_new = max(0, min(h, y_min_new))
        x_max_new = max(0, min(w, x_max_new))
        y_max_new = max(0, min(h, y_max_new))
        
        # Convert back to YOLO format (normalized center x, center y, width, height)
        new_cx = (x_min_new + x_max_new) / 2.0 / w
        new_cy = (y_min_new + y_max_new) / 2.0 / h
        new_bw = (x_max_new - x_min_new) / w
        new_bh = (y_max_new - y_min_new) / h
        
        new_line = f"{cls} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}\n"
        new_label_lines.append(new_line)
        
    # --- Photometric / Appearance Transformations ---
    # Brightness/Contrast
    if random.random() < 0.5:
        alpha = np.random.uniform(0.8, 1.2)
        beta = np.random.randint(-30, 30)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # Hue & Saturation adjustment (via HSV)
    if random.random() < 0.5:
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_delta = random.randint(-10, 10)
        s_mult = random.uniform(0.9, 1.1)
        image_hsv[:,:,0] = np.clip(image_hsv[:,:,0] + h_delta, 0, 179)
        image_hsv[:,:,1] = np.clip(image_hsv[:,:,1] * s_mult, 0, 255)
        image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    
    # Gaussian Blur
    if random.random() < 0.3:
        ksize = random.choice([3, 5])  # Kernel size should be odd
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    
    # Sharpness Enhancement
    if random.random() < 0.3:
        kernel_sharp = np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel_sharp)
    
    # Gamma Correction
    if random.random() < 0.3:
        gamma = random.uniform(0.8, 1.2)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(256)]).astype("uint8")
        image = cv2.LUT(image, table)
    
    # CLAHE (applied on LAB L-channel)
    if random.random() < 0.3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return image, new_label_lines

# %% [markdown]
# ### Apply Augmentations and Save Updated Labels

# %%
for label_file in os.listdir(labels_dir):
    label_path = os.path.join(labels_dir, label_file)
    image_file = os.path.splitext(label_file)[0] + ".jpg"
    image_path = os.path.join(images_dir, image_file)
    
    if os.path.isfile(label_path) and os.path.isfile(image_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Save the original image and label if not already saved
        orig_aug_img_path = os.path.join(augmented_images_dir, image_file)
        if not os.path.exists(orig_aug_img_path):
            cv2.imwrite(orig_aug_img_path, cv2.imread(image_path))
            
        orig_aug_label_path = os.path.join(augmented_labels_dir, label_file)
        if not os.path.exists(orig_aug_label_path):
            with open(orig_aug_label_path, 'w') as f:
                f.writelines(lines)
    
        # Determine the classes in this image
        image_classes = set(int(line.split()[0]) for line in lines)
        max_augmentations = max(augmentations_per_image[class_id] for class_id in image_classes)
    
        print(f"\nProcessing image: {image_file}")
        print(f"  Augmenting {max_augmentations} times")
    
        for i in range(max_augmentations):
            # Read image again to get the original (unaugmented) image
            img = cv2.imread(image_path)
            aug_img, aug_labels = augment_image_and_labels(img, lines)
    
            aug_img_path = os.path.join(augmented_images_dir, f"{os.path.splitext(image_file)[0]}_aug{i}.jpg")
            cv2.imwrite(aug_img_path, aug_img)
    
            aug_label_path = os.path.join(augmented_labels_dir, f"{os.path.splitext(label_file)[0]}_aug{i}.txt")
            with open(aug_label_path, 'w') as f:
                f.writelines(aug_labels)
    
            print(f"  → Augmented image saved: {aug_img_path}")
            print(f"  → Augmented label saved: {aug_label_path}")

# %%
aug_class_counts = count_class_instances(augmented_labels_dir, class_names)
print("\nFinal class distribution after augmentation:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {aug_class_counts[i]}")

# %% [markdown]
# ### Visualization Function
#
# This function randomly selects an augmented image and its corresponding label file,
# draws the bounding boxes (converted from YOLO normalized format to absolute coordinates),
# and shows the image using matplotlib.

# %%
def visualize_random_annotation(aug_images_dir, aug_labels_dir):
    # List all augmented image files and choose one randomly.
    image_files = [f for f in os.listdir(aug_images_dir) if f.endswith('.jpg')]
    if not image_files:
        print("No augmented images found!")
        return
    
    selected_image_file = random.choice(image_files)
    image_path = os.path.join(aug_images_dir, selected_image_file)
    
    # Assume the corresponding label file has the same base name with '.txt'
    label_file = os.path.splitext(selected_image_file)[0] + ".txt"
    label_path = os.path.join(aug_labels_dir, label_file)
    
    if not os.path.exists(label_path):
        print("Corresponding label file not found for", selected_image_file)
        return
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    # Draw bounding boxes
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        # convert normalized YOLO format to absolute coordinates
        xmin = int((cx - bw/2) * w)
        ymin = int((cy - bh/2) * h)
        xmax = int((cx + bw/2) * w)
        ymax = int((cy + bh/2) * h)
        
        # Choose a color based on class (random here; you can predefine colors)
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(img, class_names[cls], (xmin, max(ymin-10,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"Annotations for {selected_image_file}")
    plt.axis('off')
    plt.show()

# %%
# To visualize, simply call:
visualize_random_annotation(augmented_images_dir, augmented_labels_dir)
