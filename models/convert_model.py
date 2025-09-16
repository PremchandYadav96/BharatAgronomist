# convert_model.py

import numpy as np
import tensorflow as tf
from scipy.io import loadmat # You may need to install this: pip install scipy

# IMPORTANT: You MUST recreate the exact same model architecture here.
# This function is a placeholder. Replace it with your actual model definition.
def create_my_cnn_model(input_shape):
    """
    This function should define and return the same Keras model
    that was used to generate the weights in the .mat file.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(16, activation='softmax') # Assuming 16 output classes for Indian Pines
    ])
    return model

# --- Conversion Steps ---

# 1. Define the input shape your model expects.
# For Indian Pines, there are 200 spectral bands.
INPUT_SHAPE = (200, 1)

# 2. Create an instance of the model architecture.
model = create_my_cnn_model(input_shape=INPUT_SHAPE)
print("Model architecture created successfully.")
model.summary()

# 3. Load the weights from your .mat file.
# Make sure the .mat file is in the same directory as this script.
mat_file_path = 'trainedIndianPinesCSCNN_v2.mat'
try:
    mat_data = loadmat(mat_file_path)
    print("\n.mat file loaded. Contents:", mat_data.keys())

    # IMPORTANT: Inspect the keys printed above. You need to find the key
    # that corresponds to the model weights. It might be named 'weights', 'net',
    # 'model_weights', etc. Replace 'weights_key_in_mat_file' with the correct key.
    weights_key = 'model_weights' # <--- CHANGE THIS KEY IF NEEDED

    if weights_key in mat_data:
        weights = mat_data[weights_key]
        model.set_weights(weights)
        print(f"\nSuccessfully set weights from key '{weights_key}'.")
    else:
        # Fallback if the key isn't obvious, try to find an array that matches.
        # This is just a guess and might not work.
        print(f"Warning: Could not find key '{weights_key}'. Searching for other possible weight arrays...")
        found_weights = False
        for key, value in mat_data.items():
            if isinstance(value, np.ndarray):
                try:
                    model.set_weights(value)
                    print(f"Successfully set weights using data from key: '{key}'")
                    found_weights = True
                    break
                except Exception as e:
                    print(f"Could not set weights from key '{key}': {e}")
        if not found_weights:
            raise ValueError("Could not find a valid weight array in the .mat file.")

    # 4. Save the complete model in the .h5 format.
    output_h5_path = 'trainedIndianPinesCSCNN_v2.h5'
    model.save(output_h5_path)
    print(f"\n✅ Model successfully converted and saved to '{output_h5_path}'")

except FileNotFoundError:
    print(f"Error: The file '{mat_file_path}' was not found. "
          f"Please place it in the same directory as this conversion script.")
except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Conversion failed. Please check your model architecture and the key name for the weights in the .mat file.")