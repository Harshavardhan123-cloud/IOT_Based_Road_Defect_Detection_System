import os
import json
import logging
from tqdm import tqdm
from ultralytics import YOLO  # YOLOv5/YOLOv8 integration
from sklearn.model_selection import train_test_split


# Configure logging
log_file = 'training_log.txt'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='w')
    ]
)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define dataset directories relative to the script's location
train_dir = os.path.join(script_dir, 'generated_images')
valid_dir = os.path.join(script_dir, 'valid')
test_dir = os.path.join(script_dir, 'test')

# Check for COCO-style JSON annotations
annotations_file = os.path.join(train_dir, '_annotations_updated.coco.json')

# Parse COCO annotations
def parse_coco_annotations(annotations_path):
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    images = {img['id']: img['file_name'] for img in coco_data['images']}
    annotations = coco_data['annotations']
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    data = []
    for ann in annotations:
        image_id = ann['image_id']
        category_id = ann['category_id']
        image_path = os.path.join(train_dir, images[image_id])
        label = category_id
        if os.path.exists(image_path):  # Check if the image exists
            data.append((image_path, label))
        else:
            logging.warning(f"Image not found: {image_path}")
    return data, categories

# Load and split data
train_annotations, class_map = parse_coco_annotations(annotations_file)
train_data, val_data = train_test_split(train_annotations, test_size=0.2, random_state=42)

# Save annotations in YOLO format
def convert_to_yolo_format(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    labels_dir = os.path.join(output_dir, 'labels')
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    for image_path, label in data:
        filename = os.path.basename(image_path)
        base_name, _ = os.path.splitext(filename)
        yolo_label_path = os.path.join(labels_dir, f"{base_name}.txt")

        # Assuming label is the center x, center y, width, height, class_id (normalized)
        with open(yolo_label_path, 'w') as f:
            f.write(f"{label} 0.5 0.5 1.0 1.0\n")  # Dummy data for demonstration

        # Copy image to the YOLO directory
        dest_image_path = os.path.join(images_dir, filename)
        if not os.path.exists(dest_image_path):
            os.link(image_path, dest_image_path)

convert_to_yolo_format(train_data, 'train_yolo')
convert_to_yolo_format(val_data, 'val_yolo')

# Train YOLO model without using a YAML file
model = YOLO('yolov5s.pt')  # Load YOLOv5s model (small version for demonstration)

# Train the model   
results = model.train(
    train='train_yolo',  # Directly use the directory for training data
    val='val_yolo',  # Directly use the directory for validation data
    epochs=50,
    imgsz=640,
    batch=16,
    project='road_defect_yolo2',
    name='experiment1',
    exist_ok=True
)

# Evaluate the model
metrics = model.val(data='val_yolo')
logging.info(f"Validation metrics: {metrics}")

# Export the model to ONNX for deployment
model.export(format='onnx')
logging.info("Model exported successfully.")
import os
import json
import logging
from tqdm import tqdm
from ultralytics import YOLO  # YOLOv5/YOLOv8 integration
from sklearn.model_selection import train_test_split


# Configure logging
log_file = 'training_log.txt'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='w')
    ]
)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define dataset directories relative to the script's location
train_dir = os.path.join(script_dir, 'generated_images')
valid_dir = os.path.join(script_dir, 'valid')
test_dir = os.path.join(script_dir, 'test')

# Check for COCO-style JSON annotations
annotations_file = os.path.join(train_dir, '_annotations_updated.coco.json')

# Parse COCO annotations
def parse_coco_annotations(annotations_path):
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    images = {img['id']: img['file_name'] for img in coco_data['images']}
    annotations = coco_data['annotations']
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    data = []
    for ann in annotations:
        image_id = ann['image_id']
        category_id = ann['category_id']
        image_path = os.path.join(train_dir, images[image_id])
        label = category_id
        if os.path.exists(image_path):  # Check if the image exists
            data.append((image_path, label))
        else:
            logging.warning(f"Image not found: {image_path}")
    return data, categories

# Load and split data
train_annotations, class_map = parse_coco_annotations(annotations_file)
train_data, val_data = train_test_split(train_annotations, test_size=0.2, random_state=42)

# Save annotations in YOLO format
def convert_to_yolo_format(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    labels_dir = os.path.join(output_dir, 'labels')
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    for image_path, label in data:
        filename = os.path.basename(image_path)
        base_name, _ = os.path.splitext(filename)
        yolo_label_path = os.path.join(labels_dir, f"{base_name}.txt")

        # Assuming label is the center x, center y, width, height, class_id (normalized)
        with open(yolo_label_path, 'w') as f:
            f.write(f"{label} 0.5 0.5 1.0 1.0\n")  # Dummy data for demonstration

        # Copy image to the YOLO directory
        dest_image_path = os.path.join(images_dir, filename)
        if not os.path.exists(dest_image_path):
            os.link(image_path, dest_image_path)

convert_to_yolo_format(train_data, 'train_yolo')
convert_to_yolo_format(val_data, 'val_yolo')

# Train YOLO model without using a YAML file
model = YOLO('yolo11n.pt')  # Load YOLOv5s model (small version for demonstration)

# Train the model   
results = model.train(
    train='train_yolo',  # Directly use the directory for training data
    val='val_yolo',  # Directly use the directory for validation data
    epochs=50,
    imgsz=640,
    batch=16,
    project='road_defect_yolo2',
    name='experiment1',
    exist_ok=True
)

# Evaluate the model
metrics = model.val(data='val_yolo')
logging.info(f"Validation metrics: {metrics}")

# Export the model to ONNX for deployment
model.export(format='onnx')
logging.info("Model exported successfully.")
