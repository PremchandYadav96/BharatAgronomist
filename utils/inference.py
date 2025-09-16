import numpy as np
from .preprocess import preprocess_image, preprocess_hyperspectral

def predict_image(img_array, model):
    img = preprocess_image(img_array)
    preds = model.predict(img)
    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    return class_idx, confidence

def predict_hyperspectral(data, scaler, pca, clf, encoder):
    data_pca = preprocess_hyperspectral(data, scaler, pca)

    if clf is None:
        return "Hyperspectral classifier not loaded", None

    preds = clf.predict(data_pca)
    label = encoder.inverse_transform(preds)[0]
    return label, None
