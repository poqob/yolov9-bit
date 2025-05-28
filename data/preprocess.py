import os
import glob

def remove_class_from_labels(directory_path, class_to_remove):
    """
    Removes a specific class from all label files in the given directory
    
    Args:
        directory_path: Path to the directory containing label files
        class_to_remove: Class ID to remove (as a string)
    """
    # Get all txt files in the specified directory
    label_files = glob.glob(os.path.join(directory_path, "*.txt"))
    print(f"Found {len(label_files)} label files in {directory_path}")
    
    modified_count = 0
    
    for label_file in label_files:
        lines_to_keep = []
        file_modified = False
        
        # Read the file
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if parts and parts[0] != class_to_remove:
                        lines_to_keep.append(line)
                    else:
                        file_modified = True
        
        # Write back the file without the removed class
        if file_modified:
            with open(label_file, 'w') as f:
                for line in lines_to_keep:
                    f.write(f"{line}\n")
            modified_count += 1
    
    print(f"Removed class {class_to_remove} from {modified_count} files")


def rename_png_to_lowercase(directory_path):
    """
    Renames all PNG files (with uppercase extension) to use lowercase .png extension
    
    Args:
        directory_path: Path to the directory containing image files
    """
    # Find all files with .PNG extension (case sensitive)
    png_files = glob.glob(os.path.join(directory_path, "*.PNG"))
    print(f"Found {len(png_files)} PNG files to rename in {directory_path}")
    
    renamed_count = 0
    
    for png_file in png_files:
        # Create new filename with lowercase extension
        new_filename = png_file[:-3] + "png"
        
        # Rename the file
        os.rename(png_file, new_filename)
        renamed_count += 1
    
    print(f"Renamed {renamed_count} files from .PNG to .png")


if __name__ == "__main__":
    dataset_dir = "all"
    
    # Make sure the path is valid
    if not os.path.exists(dataset_dir):
        # Try with absolute path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        dataset_dir = os.path.join(parent_dir, "data/dataset/all")
        
        if not os.path.exists(dataset_dir):
            print(f"Error: Directory {dataset_dir} does not exist")
            exit(1)
    
    print(f"Processing files in {dataset_dir}")
    
    # Uncomment the function you want to run
    # remove_class_from_labels(dataset_dir, "2")  # Remove class 2 (artefakt)
    rename_png_to_lowercase(dataset_dir)  # Rename PNG to png
    
    print("Done!")