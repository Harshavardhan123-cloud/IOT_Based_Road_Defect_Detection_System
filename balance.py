import os
import yaml
import matplotlib.pyplot as plt


def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def count_class_instances(label_dir, num_classes):
    class_counts = [0] * num_classes
    label_map = {}

    for fname in os.listdir(label_dir):
        if fname.endswith('.txt'):
            path = os.path.join(label_dir, fname)
            with open(path, 'r') as f:
                lines = f.readlines()
            labels = []
            for line in lines:
                cls_id = int(line.strip().split()[0])
                class_counts[cls_id] += 1
                labels.append((cls_id, line.strip()))
            label_map[path] = labels
    return class_counts, label_map

def reduce_class_instances(label_map, target_class, target_remove_count):
    removed = 0
    for file, labels in list(label_map.items()):
        if removed >= target_remove_count:
            break
        new_labels = []
        for cls_id, line in labels:
            if cls_id == target_class and removed < target_remove_count:
                removed += 1
                continue  # skip this instance
            new_labels.append(line)
        if not new_labels:
            os.remove(file)  # remove annotation file if empty
            img_path = file.replace('/labels/', '/images/').replace('.txt', '.jpg')
            if os.path.exists(img_path):
                os.remove(img_path)
        else:
            with open(file, 'w') as f:
                f.write('\n'.join(new_labels) + '\n')
    print(f"Removed {removed} instances of class {target_class}")

def main():
    yaml_path = 'Aug_Fin_Dataset/data.yaml'  # path to your YAML
    data = load_yaml(yaml_path)

    class_names = data['names']
    target_class_name = 'Raveling'

    if target_class_name not in class_names:
        print(f"Class '{target_class_name}' not found in YAML.")
        return

    target_class_id = class_names.index(target_class_name)

    train_image_dir = data['train']
    label_dir = train_image_dir.replace('/images', '/labels')

    if not os.path.exists(label_dir):
        print(f"Label directory not found: {label_dir}")
        return

    class_counts, label_map = count_class_instances(label_dir, len(class_names))

    print("Class counts before deletion:")
    for i, name in enumerate(class_names):
        print(f"{name}: {class_counts[i]}")

    
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_counts, color='skyblue')
    plt.xlabel('Class Names')
    plt.ylabel('Number of Instances')
    plt.title('Instance Count per Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Balancing logic
    min_other_count = min([count for i, count in enumerate(class_counts) if i != target_class_id])
    excess = class_counts[target_class_id] - min_other_count

    if excess > 0:
        reduce_class_instances(label_map, target_class_id, excess)
    else:
        print(f"No excess instances of '{target_class_name}' to remove.")

if __name__ == "__main__":
    main()
