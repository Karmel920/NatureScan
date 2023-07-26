import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from PIL import Image as PILImage
from io import BytesIO
import wikipediaapi

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(ROOT_DIR, './models/')

wiki_wiki = wikipediaapi.Wikipedia('NatureScan https://github.com/Karmel920/NatureScan','en')

flower_categories = ['crocus', 'bellis', 'dandelion', 'pansy', 'rose', 'snowdrop', 'sunflower', 'tulip']


def preprocess_image(image):
    image_data = image.read()
    image_pred = PILImage.open(BytesIO(image_data))
    image_pred = image_pred.resize((224, 224))
    image_pred = np.array(image_pred)
    image_pred = np.expand_dims(image_pred, axis=0)

    return image_pred


def get_description(flower_name):
    description = ''
    page_pred = wiki_wiki.page(flower_name)
    if page_pred.exists():
        description = page_pred.summary

    return description


def load_model_pred():
    path = os.path.join(MODELS_PATH, 'flowers_8_small_data_hq.h5')
    path = path.replace(os.sep, '/')
    load = load_model(path)
    return load


model = load_model_pred()


def make_prediction(image):
    global model

    if model is None:
        model = load_model_pred()

    image = preprocess_image(image)
    predicted = model.predict(image).tolist()[0]
    predicted_label = np.argmax(predicted)
    pred_class = flower_categories[predicted_label]
    description = get_description(pred_class)
    data = {"predicted_class": pred_class, "description": description}

    return data
