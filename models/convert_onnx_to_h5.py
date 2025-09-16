# convert_onnx_to_h5.py

import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import os

try:
    # Define the file paths
    onnx_model_path = 'trainedIndianPinesCSCNN.onnx'
    h5_model_path = 'trainedIndianPinesCSCNN.h5'
    
    print(f"Loading ONNX model from: {onnx_model_path}")
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"The ONNX file was not found at '{onnx_model_path}'. "
                              "Please make sure this script is in the same directory as the model.")

    # Load the ONNX model
    onnx_model = onnx.load(onnx_model_path)
    
    print("Preparing to convert ONNX model to TensorFlow...")
    # Prepare the TensorFlow representation of the ONNX model
    tf_rep = prepare(onnx_model)
    
    # --- Exporting as SavedModel first is the most reliable path ---
    saved_model_dir = 'temp_saved_model'
    print(f"Exporting TensorFlow representation to SavedModel format at '{saved_model_dir}'...")
    tf_rep.export_graph(saved_model_dir)
    
    # --- Now, load the SavedModel as a Keras model and save as H5 ---
    print(f"Loading model from SavedModel directory...")
    keras_model = tf.keras.models.load_model(saved_model_dir)
    
    print(f"Saving Keras model to .h5 format at '{h5_model_path}'...")
    keras_model.save(h5_model_path)
    
    print("\n✅ Conversion successful!")
    print(f"Your Keras model is now available at: {h5_model_path}")

except Exception as e:
    print(f"\n❌ An error occurred during conversion: {e}")
    print("Please ensure you have installed the required libraries: pip install onnx onnx-tf tensorflow")
