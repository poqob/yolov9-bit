#!/usr/bin/env python3
# Dataset Analysis Script for YOLOv9

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import seaborn as sns
from pathlib import Path

# Set the data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'all')

def load_classes():
    """Load class names from classes.txt"""
    classes_file = os.path.join(os.path.dirname(DATA_DIR), 'classes.txt')
    with open(classes_file, 'r') as f:
        # Skip first line if it's a comment/filepath
        lines = f.readlines()
        if lines[0].startswith('//'):
            lines = lines[1:]
        return [line.strip() for line in lines]

def analyze_directory(directory_path, classes, output_prefix=""):
    """Analyze a specific directory and return statistics"""
    stats_text = [f"===== Analysis for {os.path.basename(directory_path)} =====\n"]
    
    # Get all image paths
    image_paths = sorted(glob.glob(os.path.join(directory_path, '*.png')))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(directory_path, '*.jpg')))
    
    if not image_paths:
        stats_text.append("No images found in the directory!")
        return stats_text, None, None, None, None, None
    
    # Count total images
    total_images = len(image_paths)
    stats_text.append(f"Total images: {total_images}")
    
    # Initialize counters
    class_counts = Counter()
    image_sizes = []
    boxes_per_image = []
    box_sizes = []
    aspect_ratios = []
    
    # Count annotations without images and images without annotations
    missing_annotations = []
    missing_images = []
    
    # Analyze each image and its annotation
    for img_path in image_paths:
        # Get corresponding annotation file
        ann_path = os.path.splitext(img_path)[0] + '.txt'
        
        # Check if annotation file exists
        if not os.path.exists(ann_path):
            missing_annotations.append(os.path.basename(img_path))
            continue
        
        # Get image dimensions
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
                image_sizes.append((img_width, img_height))
        except Exception as e:
            stats_text.append(f"Error reading image {img_path}: {e}")
            continue
        
        # Read annotations
        with open(ann_path, 'r') as f:
            lines = f.readlines()
            if lines and lines[0].startswith('//'):
                lines = lines[1:]  # Skip comment line if present
            
            boxes = []
            for line in lines:
                try:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # YOLO format: class x_center y_center width height
                        class_id = int(parts[0])
                        if class_id < len(classes):
                            class_counts[classes[class_id]] += 1
                            
                            # Calculate box dimensions in pixels
                            x_center, y_center = float(parts[1]), float(parts[2])
                            width, height = float(parts[3]), float(parts[4])
                            
                            # Convert relative coordinates to absolute pixels
                            abs_width = width * img_width
                            abs_height = height * img_height
                            
                            box_sizes.append(abs_width * abs_height)
                            aspect_ratios.append(abs_width / abs_height if abs_height > 0 else 0)
                            boxes.append((class_id, x_center, y_center, width, height))
                except Exception as e:
                    stats_text.append(f"Error parsing annotation in {ann_path}: {e}")
            
            boxes_per_image.append(len(boxes))
    
    # Check for annotation files without corresponding images
    ann_paths = glob.glob(os.path.join(directory_path, '*.txt'))
    for ann_path in ann_paths:
        if ann_path.endswith('.txt'):
            img_path_base = os.path.splitext(ann_path)[0]
            if not os.path.exists(img_path_base + '.png') and not os.path.exists(img_path_base + '.jpg'):
                missing_images.append(os.path.basename(ann_path))
    
    # Print basic statistics
    stats_text.append(f"\nDistribution of classes:")
    for cls, count in class_counts.items():
        class_percentage = count/sum(class_counts.values()) if sum(class_counts.values()) > 0 else 0
        stats_text.append(f"  - {cls}: {count} annotations ({class_percentage:.2%})")
    
    # Image statistics
    if image_sizes:
        avg_width = sum(w for w, h in image_sizes) / len(image_sizes)
        avg_height = sum(h for w, h in image_sizes) / len(image_sizes)
        min_width = min(w for w, h in image_sizes)
        min_height = min(h for w, h in image_sizes)
        max_width = max(w for w, h in image_sizes)
        max_height = max(h for w, h in image_sizes)
        
        stats_text.append(f"\nImage size statistics:")
        stats_text.append(f"  - Average dimensions: {avg_width:.1f} x {avg_height:.1f}")
        stats_text.append(f"  - Minimum dimensions: {min_width} x {min_height}")
        stats_text.append(f"  - Maximum dimensions: {max_width} x {max_height}")
    
    # Bounding box statistics
    if boxes_per_image:
        total_boxes = sum(boxes_per_image)
        avg_boxes = np.mean(boxes_per_image)
        max_boxes = max(boxes_per_image)
        no_boxes_count = boxes_per_image.count(0)
        
        stats_text.append(f"\nBounding box statistics:")
        stats_text.append(f"  - Total bounding boxes: {total_boxes}")
        stats_text.append(f"  - Average boxes per image: {avg_boxes:.2f}")
        stats_text.append(f"  - Maximum boxes in an image: {max_boxes}")
        stats_text.append(f"  - Images without any boxes: {no_boxes_count}")
    
    # Missing files
    if missing_annotations:
        stats_text.append(f"\nWarning: {len(missing_annotations)} images without annotation files")
        if len(missing_annotations) <= 10:  # Show details if not too many
            for file in missing_annotations:
                stats_text.append(f"  - {file}")
    
    if missing_images:
        stats_text.append(f"\nWarning: {len(missing_images)} annotation files without corresponding images")
        if len(missing_images) <= 10:  # Show details if not too many
            for file in missing_images:
                stats_text.append(f"  - {file}")
    
    # Create visualization directory
    vis_dir = os.path.join(os.path.dirname(DATA_DIR), 'analysis_results')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Visualize class distribution
    if class_counts:
        plt.figure(figsize=(10, 6))
        plt.bar(class_counts.keys(), class_counts.values())
        plt.title(f'Class Distribution - {output_prefix}')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'{output_prefix}class_distribution.png'))
        stats_text.append(f"\nSaved class distribution plot to {os.path.join(vis_dir, f'{output_prefix}class_distribution.png')}")
    
    # Visualize boxes per image
    if boxes_per_image:
        plt.figure(figsize=(10, 6))
        plt.hist(boxes_per_image, bins=20)
        plt.title(f'Objects per Image - {output_prefix}')
        plt.xlabel('Number of Objects')
        plt.ylabel('Number of Images')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'{output_prefix}objects_per_image.png'))
        stats_text.append(f"Saved objects per image plot to {os.path.join(vis_dir, f'{output_prefix}objects_per_image.png')}")
    
    return stats_text, class_counts, boxes_per_image, box_sizes, aspect_ratios, total_images

def analyze_dataset():
    """Analyze all dataset directories and print statistics"""
    print("\n===== YOLOv9 Dataset Analysis =====\n")
    
    # For storing statistics to write to file later
    all_stats_text = ["===== YOLOv9 Dataset Analysis =====\n"]
    
    # Load class names
    classes = load_classes()
    print(f"Classes: {classes}")
    print(f"Number of classes: {len(classes)}")
    all_stats_text.append(f"Classes: {classes}")
    all_stats_text.append(f"Number of classes: {len(classes)}")
    
    # Define directories to analyze
    base_dir = os.path.dirname(DATA_DIR)
    directories = {
        'all': DATA_DIR,
        'train': os.path.join(base_dir, 'train'),
        'val': os.path.join(base_dir, 'val'),
        'test': os.path.join(base_dir, 'test')
    }
    
    # Store results for summary
    directory_stats = {}
    
    # Analyze each directory
    for dir_name, dir_path in directories.items():
        if os.path.exists(dir_path):
            print(f"\n----- Analyzing {dir_name} directory -----")
            stats_text, class_counts, boxes_per_image, box_sizes, aspect_ratios, total_images = analyze_directory(
                dir_path, classes, f"{dir_name}_"
            )
            
            # Print stats for this directory
            for line in stats_text:
                print(line)
            
            # Add to overall stats
            all_stats_text.extend(stats_text)
            
            # Store for summary
            if total_images is not None:
                directory_stats[dir_name] = {
                    'images': total_images,
                    'annotations': sum(boxes_per_image) if boxes_per_image else 0
                }
        else:
            print(f"\nDirectory {dir_name} not found at {dir_path}")
            all_stats_text.append(f"\nDirectory {dir_name} not found at {dir_path}")
    
    # Create visualization directory
    vis_dir = os.path.join(os.path.dirname(DATA_DIR), 'analysis_results')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create summary chart for images and annotations
    if directory_stats:
        plt.figure(figsize=(12, 6))
        
        # Data for plotting
        dirs = list(directory_stats.keys())
        images = [directory_stats[d]['images'] for d in dirs]
        annotations = [directory_stats[d]['annotations'] for d in dirs]
        
        # Create bar positions
        x = np.arange(len(dirs))
        width = 0.35
        
        # Create bars
        plt.bar(x - width/2, images, width, label='Görüntüler')
        plt.bar(x + width/2, annotations, width, label='Etiketler')
        
        # Add labels and title
        plt.xlabel('Veri Seti Bölümleri')
        plt.ylabel('Sayı')
        plt.title('Veri Seti Bölümlerine Göre Görüntü ve Etiket Sayıları')
        plt.xticks(x, dirs)
        plt.legend()
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'dataset_summary.png'))
        print(f"\nSaved dataset summary plot to {os.path.join(vis_dir, 'dataset_summary.png')}")
        all_stats_text.append(f"\nSaved dataset summary plot to {os.path.join(vis_dir, 'dataset_summary.png')}")
    
    # Save all statistics to a text file
    stats_file_path = os.path.join(vis_dir, 'dataset_statistics.txt')
    with open(stats_file_path, 'w') as f:
        f.write('\n'.join(all_stats_text))
    print(f"\nSaved all statistics to {stats_file_path}")
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    analyze_dataset()