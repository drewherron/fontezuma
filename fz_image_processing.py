import argparse
import cv2
import os
import numpy as np

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No image found at {image_path}")
    return image

def save_image(image, export_dir, filename):
    if not filename.endswith('.png') and not filename.endswith('.jpg'):
        filename += '.png'

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    export_path = os.path.join(export_dir, filename)
    cv2.imwrite(export_path, image)
    print(f"Image saved to {export_path}")

def detect_and_normalize(image_path, export, export_dir='export', resize_dims=(200, 200)):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_box_img = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(bounding_box_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    if export:
        save_image(bounding_box_img, export_dir, 'bounding_boxes.png')

    char_images = []
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        char = binary[y:y+h, x:x+w]
        char_resized = cv2.resize(char, resize_dims)
        char_inverted = cv2.bitwise_not(char_resized)
        char_images.append(char_inverted)
        if export:
            save_image(char_inverted, export_dir, f'char_{idx}.png')

    return char_images


# For testing directly
if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    char_imgs = detect_and_normalize(path)
    print(f"Detected {len(char_imgs)} characters.")
