import json
import sys
import cv2
import numpy as np
from collections import defaultdict
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


model_path = "font_recognition_model.keras"
indices_path = "class_indices.json"
model = load_model(model_path)
with open(indices_path, 'r') as json_file:
    class_indices = json.load(json_file)
class_labels = {v: k for k, v in class_indices.items()}

# Use the model and indices to make a prediction
def predict_font(image, model_path="font_recognition_model.keras", indices_path="class_indices.json"):
    if isinstance(image, str):
        image = load_img(image, target_size=(64, 64), color_mode='rgb')
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        image = cv2.resize(image, (64, 64))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)

    # Predict and label the predictions
    prediction = model.predict(image, verbose=False)[0]
    class_labels = {v: k for k, v in class_indices.items()}
    labeled_predictions = [(class_labels[i], float(prob)) for i, prob in enumerate(prediction)]
    labeled_predictions.sort(key=lambda x: x[1], reverse=True)

    return labeled_predictions

# I have these parallel "file" functions because they worked better when running this file directly
def predict_font_from_file(image_path, model_path="font_recognition_model.keras", indices_path="class_indices.json"):
    # Load and preprocess the image
    image = load_img(image_path, target_size=(64, 64), color_mode='rgb')
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0

    # Predict and label the predictions
    prediction = model.predict(image)[0]
    class_labels = {v: k for k, v in class_indices.items()}
    labeled_predictions = [(class_labels[i], float(prob)) for i, prob in enumerate(prediction)]
    labeled_predictions.sort(key=lambda x: x[1], reverse=True)

    return labeled_predictions

# Combine the scores of each font
def aggregate_predictions(predictions):
    font_scores = defaultdict(float)
    for prediction in predictions:
        for font, score in prediction:
            font_scores[font] += score

    # Sort and return the top results
    sorted_fonts = sorted(font_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_fonts

# Aggregate scores for this file's main()
def aggregate_file_predictions(image_paths):
    predictions = []
    for image_path in image_paths:
        result = predict_font_from_file(image_path)
        predictions.extend(result)

    font_scores = {}
    for font, score in predictions:
        if font in font_scores:
            font_scores[font] += score
        else:
            font_scores[font] = score

    # Sort the results
    sorted_fonts = sorted(font_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_fonts


# For testing directly
def main():
    if len(sys.argv) < 2:
        print("Usage: python fz_predict_font.py <path_to_image1> <path_to_image2> ...")
        sys.exit(1)

    image_paths = sys.argv[1:]
    agg_predictions = aggregate_file_predictions(image_paths)

    print("Font predictions:")
    for font, score in agg_predictions[:5]:
        print(f"{font}: {score:.4f}")

if __name__ == '__main__':
    main()
