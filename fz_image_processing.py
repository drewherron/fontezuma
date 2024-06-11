import argparse
import cv2
import os
import numpy as np

def load_image(image_path):
    if not image_path or not isinstance(image_path, str):
        raise ValueError("The image_path must be a valid string.")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No image found at {image_path}, or the path is incorrect.")
    return image

def save_image(image, export_dir, filename, verbose=False):
    if not filename.endswith('.png') and not filename.endswith('.jpg'):
        filename += '.png'

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    export_path = os.path.join(export_dir, filename)
    cv2.imwrite(export_path, image)
    if verbose:
        print(f"Image saved to {export_path}")

def preprocess(image):
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    # Noise reduction
    #image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    #image = cv2.GaussianBlur(image, (5, 5), 0)
    # Apply OTSU thresholding
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Morphological erosion
    #kernel = np.ones((10, 10), np.uint8)
    #image = cv2.erode(image, kernel, iterations=1)
    return image

def detect_and_normalize(image_path, export, verbose, export_dir='export', resize_dims=(200, 200)):
    # Load the image
    img = load_image(image_path)
    # Process image
    processed_image = preprocess(img)

    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_box_img = cv2.cvtColor(processed_image.copy(), cv2.COLOR_GRAY2BGR)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(bounding_box_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    if export:
        save_image(bounding_box_img, export_dir, 'bounding_boxes.png', verbose)

    char_images = []
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        char = processed_image[y:y+h, x:x+w]
        char_resized = cv2.resize(char, resize_dims)
        char_inverted = cv2.bitwise_not(char_resized)
        char_images.append(char_inverted)
        if export:
            save_image(char_inverted, export_dir, f'char_{idx}.png', verbose)

    return char_images

# For testing directly
if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    char_imgs = detect_and_normalize(path)
    print(f"Detected {len(char_imgs)} characters.")
