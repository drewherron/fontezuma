import json
import sys
import numpy as np
from collections import defaultdict
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_class_indices(path="class_indices.json"):
    import json
    with open(path, 'r') as json_file:
        class_indices = json.load(json_file)
    return class_indices

def predict_font(image_path, model_path="font_recognition_model.keras", indices_path="class_indices.json"):
    model = load_model(model_path)
    class_indices = load_class_indices(indices_path)

    img = load_img(image_path, target_size=(64, 64))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)[0]
    class_labels = {v: k for k, v in class_indices.items()}
    labeled_predictions = [(class_labels[i], float(prob)) for i, prob in enumerate(prediction)]
    labeled_predictions.sort(key=lambda x: x[1], reverse=True)

    # Return the top 5 predictions
    return labeled_predictions[:5]

def aggregate_predictions(image_paths, model_path="font_recognition_model.keras", indices_path="class_indices.json"):
    font_scores = defaultdict(float)

    for image_path in image_paths:
        predictions = predict_font(image_path, model_path, indices_path)
        for font, prob in predictions:
            # Weighted voting
            font_scores[font] += prob

    # Sort fonts by their aggregated scores
    sorted_fonts = sorted(font_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_fonts
