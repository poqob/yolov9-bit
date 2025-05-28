import os
import shutil
import random
from glob import glob

def prepare_yolo_dataset(all_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data from all_dir into train, validation, and test sets for YOLOv9 training.
    
    Args:
        all_dir: Directory containing all images and label files
        train_dir: Directory for training data
        val_dir: Directory for validation data
        test_dir: Directory for test data
        train_ratio: Ratio of data to use for training (default: 0.7)
        val_ratio: Ratio of data to use for validation (default: 0.15)
        test_ratio: Ratio of data to use for testing (default: 0.15)
    """
    # Check if ratios sum to 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        raise ValueError("Train, validation, and test ratios must sum to 1")

    # Create directories if they don't exist
    for directory in [train_dir, val_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            # Clear existing files
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Cleared directory: {directory}")

    # Get all image files (png format)
    image_files = sorted(glob(os.path.join(all_dir, "*.png")))
    
    if not image_files:
        print(f"No image files found in {all_dir}")
        return

    print(f"Found {len(image_files)} images in total")

    # Shuffle the image files for random split
    random.shuffle(image_files)
    
    # Calculate split points
    train_end = int(len(image_files) * train_ratio)
    val_end = train_end + int(len(image_files) * val_ratio)
    
    # Split the data
    train_images = image_files[:train_end]
    val_images = image_files[train_end:val_end]
    test_images = image_files[val_end:]
    
    print(f"Split distribution: {len(train_images)} train, {len(val_images)} validation, {len(test_images)} test")
    
    # Copy files to their respective directories
    for img_set, target_dir in [(train_images, train_dir), (val_images, val_dir), (test_images, test_dir)]:
        for img_path in img_set:
            # Get the base filename
            base_name = os.path.basename(img_path)
            img_name = base_name
            label_name = os.path.splitext(base_name)[0] + ".txt"
            
            # Source paths
            img_src = img_path
            label_src = os.path.join(all_dir, label_name)
            
            # Target paths
            img_dst = os.path.join(target_dir, img_name)
            label_dst = os.path.join(target_dir, label_name)
            
            # Copy image file
            shutil.copy2(img_src, img_dst)
            
            # Copy label file if it exists
            if os.path.exists(label_src):
                shutil.copy2(label_src, label_dst)
            else:
                print(f"Warning: Label file not found for {img_name}")

    print("Dataset preparation completed successfully!")

def update_dataset_yaml(yaml_path, num_classes, class_names):
    """
    Update the dataset.yaml file with correct paths and class information.
    
    Args:
        yaml_path: Path to the dataset.yaml file
        num_classes: Number of classes in the dataset
        class_names: List of class names
    """
    # Define the content for the YAML file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_content = f"""# YOLOv9 dataset configuration
path: {current_dir}  # dataset root directory
train: train  # train images relative to 'path'
val: val  # validation images relative to 'path'
test: test  # test images relative to 'path'

# Classes
nc: {num_classes}  # number of classes
names: {class_names}  # class names
"""

    # Write the content to the YAML file
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Updated YAML configuration file: {yaml_path}")

def read_class_names(classes_file):
    """
    Read class names from the classes.txt file.
    
    Args:
        classes_file: Path to the classes.txt file
    
    Returns:
        List of class names
    """
    try:
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        return class_names
    except FileNotFoundError:
        print(f"Warning: Class file {classes_file} not found. Using default class ['ECG']")
        return ['ECG']

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define directories
    data_dir = os.path.dirname(os.path.abspath(__file__))
    all_dir = os.path.join(data_dir, "all")
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    
    # Prepare the dataset
    prepare_yolo_dataset(all_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Read class names from classes.txt
    classes_file = os.path.join(data_dir, "classes.txt")
    class_names = read_class_names(classes_file)
    num_classes = len(class_names)
    
    # Update the dataset.yaml file
    yaml_path = os.path.join(data_dir, "dataset.yaml")
    update_dataset_yaml(yaml_path, num_classes, class_names)
    
    print("Dataset preparation and YAML configuration completed successfully!")
    print(f"Training set: {len(glob(os.path.join(train_dir, '*.png')))} images")
    print(f"Validation set: {len(glob(os.path.join(val_dir, '*.png')))} images")
    print(f"Testing set: {len(glob(os.path.join(test_dir, '*.png')))} images")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")