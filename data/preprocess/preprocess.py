# filepath: /mnt/newdisk/dosyalar/Dosyalar/projeler/py/yolov9-bit/data/preprocess/preprocess.py
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def colorize_to_ekg(input_img_path, output_img_path=None, color=(204, 105, 255)):
    """
    Colorize an image to look like an EKG with a reddish-pink color.
    
    Args:
        input_img_path (str): Path to the input image
        output_img_path (str, optional): Path to save the output image. If None, the output will be shown.
        color (tuple): The BGR color value to use (default is reddish-pink)
    """
    # Read the image
    img = cv2.imread(input_img_path)
    
    if img is None:
        print(f"Error: Could not read image at {input_img_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create a blank colored image (white background)
    colored = np.ones_like(img) * 255
    
    # Get a binary mask of the darker parts of the image (adjust threshold as needed)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Apply the color to the mask
    colored[thresh > 0] = color
    
    # Optional: Add some blending to make it look more natural
    alpha = 0.8
    blended = cv2.addWeighted(img, 1-alpha, colored, alpha, 0)
    
    # Save or display the result
    if output_img_path:
        cv2.imwrite(output_img_path, blended)
        print(f"Colorized image saved to {output_img_path}")
    else:
        cv2.imshow("Colorized EKG", blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return blended

def preprocess_ekg_image(input_img_path, output_dir=None):
    """
    Apply various preprocessing steps to an EKG image and save the outputs
    
    Args:
        input_img_path (str): Path to the input image
        output_dir (str, optional): Directory to save the output images. If None, uses the directory of the input image.
    """
    # Read the image
    img = cv2.imread(input_img_path)
    
    if img is None:
        print(f"Error: Could not read image at {input_img_path}")
        return
    
    # Create output directory if needed
    if output_dir is None:
        output_dir = os.path.dirname(input_img_path)
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.splitext(os.path.basename(input_img_path))[0]
    
    # Save original image
    original_output = os.path.join(output_dir, f"{base_filename}_a_original.jpg")
    cv2.imwrite(original_output, img)
    print(f"Original image saved to {original_output}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_output = os.path.join(output_dir, f"{base_filename}_b_grayscale.jpg")
    cv2.imwrite(gray_output, gray)
    print(f"Grayscale image saved to {gray_output}")
    
    # Enhance contrast using histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)
    contrast_output = os.path.join(output_dir, f"{base_filename}_c_contrast_enhanced.jpg")
    cv2.imwrite(contrast_output, contrast_enhanced)
    print(f"Contrast enhanced image saved to {contrast_output}")
    
    # Create a figure to display all preprocessing steps
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('(a) Orijinal Görüntü')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('(b) Gri Tonlamalı Görüntü')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(contrast_enhanced, cmap='gray')
    plt.title('(c) Kontrast İyileştirilmiş Görüntü')
    plt.axis('off')
    
    plt_output = os.path.join(output_dir, f"{base_filename}_preprocessing_steps.jpg")
    plt.tight_layout()
    plt.savefig(plt_output)
    print(f"Visualization of preprocessing steps saved to {plt_output}")
    
    return {
        'original': img,
        'grayscale': gray,
        'contrast_enhanced': contrast_enhanced
    }

if __name__ == "__main__":
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set the input and output paths
    input_path = os.path.join(script_dir, "example.jpg")
    output_dir = os.path.join(script_dir, "preprocessing_results")
    
    # Apply preprocessing steps
    preprocess_ekg_image(input_path, output_dir)
    
    # Also apply colorization as before
    output_path = os.path.join(script_dir, "example_ekg_colored.jpg")
    ekg_color = (150, 80, 220)  # BGR value for reddish-pink
    colorize_to_ekg(input_path, output_path, ekg_color)