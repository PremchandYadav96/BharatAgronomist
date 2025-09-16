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

def run_hyperspectral_analysis_onnx(hyperspectral_model, spectral_data_input):
    """
    Runs hyperspectral analysis using the ONNX model.
    Returns the predicted label and confidence.
    """
    try:
        spectral_values = [float(val.strip()) for val in spectral_data_input.split(',')]
        num_bands = len(spectral_values)
        data_reshaped = np.array(spectral_values).reshape(1, 1, 1, num_bands, 1)
        data_final = data_reshaped.astype(np.float32)

        input_name = hyperspectral_model.get_inputs()[0].name
        output_name = hyperspectral_model.get_outputs()[0].name
        result = hyperspectral_model.run([output_name], {input_name: data_final})

        prediction_array = result[0]
        predicted_class_index = np.argmax(prediction_array, axis=1)[0]

        # Using softmax to get confidence
        import tensorflow as tf
        softmax_scores = tf.nn.softmax(prediction_array[0]).numpy()
        confidence = np.max(softmax_scores) * 100

        class_labels = [
            'Alfalfa', 'Corn-notill', 'Corn-min', 'Corn', 'Grass-pasture',
            'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats',
            'Soybean-notill', 'Soybean-min', 'Soybean-clean', 'Wheat', 'Woods',
            'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers'
        ]

        if predicted_class_index < len(class_labels):
            predicted_label = class_labels[predicted_class_index]
            return predicted_label, confidence
        else:
            return "Prediction index out of bounds", None
    except Exception as e:
        return f"An error occurred during analysis: {e}", None
