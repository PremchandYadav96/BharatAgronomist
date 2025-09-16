import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

def process_and_display_prediction(image_file, plant_model):
    """Takes an image file, runs prediction, and displays results."""
    if plant_model is None:
        st.error("The plant disease prediction model is currently unavailable.")
        return
    st.image(image_file, caption="Uploaded Leaf Image", use_column_width=True)
    with st.spinner("🔬 Analyzing image..."):
        image = Image.open(image_file).convert("RGB")
        img_resized = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = plant_model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100
        # NOTE: This is a best-effort list of the 38 classes from the PlantVillage dataset.
        class_labels = [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Blueberry___healthy',
            'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Raspberry___healthy',
            'Soybean___healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        predicted_class_label = class_labels[predicted_class_index]
        st.success("Analysis Complete!")
        if "healthy" in predicted_class_label: st.metric("Prediction Result", "✅ Healthy")
        else: st.metric("Prediction Result", "❌ Disease Detected")
        st.subheader("Detected Condition")
        st.write(f"**{predicted_class_label.replace('___', ' - ').replace('_', ' ')}**")
        st.progress(int(confidence))
        st.caption(f"Confidence: {confidence:.2f}%")
        with st.expander("🔬 View Management & Prevention Tips"):
            st.info("Prune affected areas, ensure proper air circulation, and use recommended organic fungicides. Avoid overhead watering.")
            st.warning("Plant disease-resistant varieties, practice crop rotation, and maintain good field sanitation.")
