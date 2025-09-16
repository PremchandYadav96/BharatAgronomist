import tensorflow as tf
import numpy as np

def preprocess_image(img_array):
    """
    Preprocess image for CNN model
    img_array: numpy array (H, W, C)
    """
    img = tf.image.resize(img_array, (224, 224)) / 255.0
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img

def preprocess_hyperspectral(data, scaler, pca):
    """
    Preprocess hyperspectral data (1D array of reflectance values)
    """
    data_scaled = scaler.transform([data])
    data_pca = pca.transform(data_scaled)
    return data_pca
