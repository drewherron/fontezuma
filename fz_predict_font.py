import json
import sys
import numpy as np
from collections import defaultdict
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def predict_font(image_path, model_path='font_recognition_model.keras'):
    model = load_model(model_path)

    img = load_img(image_path, target_size=(64, 64))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # labeled_predictions = yadda yadda yadda

    return labeled_predictions

