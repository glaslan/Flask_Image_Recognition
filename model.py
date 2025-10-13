'''
This module handles image preprocessing and digit prediction using a trained Keras model.
'''

# Importing required libs
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image

# Loading model
model = load_model("digit_model.h5")


# Preparing and pre-processing the image
def preprocess_img(img_path):
    """
    Preprocess an image for model prediction.

    Args:
        img_path: File path or stream to the image.

    Returns:
        numpy.ndarray: Preprocessed image array of shape (1, 224, 224, 3).
    """
    op_img = Image.open(img_path)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape


# Predicting function
def predict_result(predict):
    """
    Predict the digit from a preprocessed image.

    Args:
        predict (numpy.ndarray): Preprocessed image array.

    Returns:
        int: Predicted digit (0-9).
    """
    pred = model.predict(predict)
    return np.argmax(pred[0], axis=-1)
