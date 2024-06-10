import argparse
import cv2
import numpy as np

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No image found at {image_path}")
    return image

def segment_characters(line_images):
    character_images = []
    for line_image in line_images:
        gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                x, y, w, h = cv2.boundingRect(contour)
                char_img = line_image[y:y+h, x:x+w]
                character_images.append(char_img)

    return character_images

def detect_and_normalize(image_path, resize_dims=(200, 200)):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        char = binary[y:y+h, x:x+w]
        char_resized = cv2.resize(char, resize_dims)
        char_inverted = cv2.bitwise_not(char_resized)
        char_images.append(char_inverted)

    return char_images


# For testing directly
if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    char_imgs = detect_and_normalize(path)
    print(f"Detected {len(char_imgs)} characters.")
