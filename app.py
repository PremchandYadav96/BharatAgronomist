# ====================================================================
# Smart Agriculture Dashboard for SIH Project
# - UI with farmer-friendly, step-by-step layout.
# - Centralized geocoding and AI-powered soil/irrigation suggestions.
# - **INSECURE API KEY HANDLING (FOR DEMO ONLY)**
# - Generates a combined Climate Analysis and Crop Advisory report.
# ====================================================================
from tensorflow.keras.models import load_model
import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import google.generativeai as genai
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import onnxruntime as ort
from nasa_power import fetch_nasa_power_data # Ensure nasa_power.py is in the same directory

# ====================================================================
# Page Configuration & API Setup
# ====================================================================
st.set_page_config(
    page_title="Smart Agriculture AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ####################################################################
# # --- WARNING: Hardcoding API keys is insecure and NOT recommended ---
# # This is for demonstration ONLY. Use st.secrets for deployment.
# ####################################################################
GEMINI_API_KEY = "AIzaSyCzghG5GiSTX6MnzaxeGXlEuh6aFCec37A" # <-- YOUR KEY IS PLACED HERE

# --- Configure Gemini API ---
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}")
        GEMINI_API_KEY = None
else:
    st.warning("⚠️ Gemini API key is not set. AI features will be disabled.")
# ####################################################################

# ====================================================================
# Model Loading (with Streamlit caching)
# ====================================================================
MODEL_DIR = os.path.join(os.getcwd(), "models")

@st.cache_resource
def load_plant_model():
    model_path = os.path.join(MODEL_DIR, "plant_disease_model.h5")
    if not os.path.exists(model_path): return None
    try:
        model = load_model(model_path)
        print("✅ Plant disease model loaded")
        return model
    except Exception as e:
        print(f"⚠️ Error loading plant disease model: {e}")
        return None

@st.cache_resource
def load_hyperspectral_model():
    model_path = os.path.join(MODEL_DIR, "trainedIndianPinesCSCNN.onnx")
    if not os.path.exists(model_path): return None
    try:
        session = ort.InferenceSession(model_path)
        print("✅ Hyperspectral ONNX model loaded")
        return session
    except Exception as e:
        print(f"⚠️ Error loading hyperspectral ONNX model: {e}")
        return None

plant_model = load_plant_model()
hyperspectral_model = load_hyperspectral_model()

# ====================================================================
# Helper Functions
# ====================================================================
@st.cache_data
def get_coords_from_place_name(place_name):
    """Converts a place name to latitude and longitude."""
    if not place_name: return None, None
    geolocator = Nominatim(user_agent="smart_agri_dashboard")
    try:
        location = geolocator.geocode(place_name)
        if location:
            return location.latitude, location.longitude
    except (GeocoderTimedOut, GeocoderUnavailable):
        st.error("Geocoding service is unavailable. Please try again later.")
    return None, None

def get_soil_and_irrigation_suggestion(location_name):
    """Suggests soil and irrigation based on location using Gemini AI."""
    if not GEMINI_API_KEY: return None, None
    if not location_name: return None, None
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = f"""
    Based on the agricultural region of '{location_name}' in India, what is the most common soil type and the predominant irrigation method?
    Provide the answer in a single, parsable line with the following exact format:
    Soil: [Primary Soil Type], Irrigation: [Primary Irrigation Method]
    Example:
    Soil: Alluvial Soil, Irrigation: Good (Drip/Sprinkler/Canal)
    Soil: Black (Regur) Soil, Irrigation: Rain-fed (No irrigation)
    """
    try:
        response = model.generate_content(prompt)
        parts = response.text.strip().split(',')
        soil = parts[0].replace('Soil:', '').strip()
        irrigation = parts[1].replace('Irrigation:', '').strip()
        return soil, irrigation
    except Exception as e:
        print(f"Error suggesting soil/irrigation: {e}")
        st.error("AI could not determine the soil/irrigation type. Please select them manually.")
        return None, None

def generate_combined_report(lat, lon, soil_type, irrigation, farm_size, weather_df, location_name):
    """Generates a combined climate and crop report using Gemini AI."""
    if not GEMINI_API_KEY: return "Cannot generate report: Gemini API key is not configured."
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    weather_summary = weather_df.describe().to_string()
    recent_weather = weather_df.tail(15).to_string()
    prompt = f"""
    **Role:** You are an expert agronomist preparing a comprehensive report for a farmer in India.
    **Farmer's Context:**
    - **Location Name:** {location_name}
    - **Coordinates:** Latitude={lat}, Longitude={lon}
    - **Soil Type:** {soil_type}
    - **Irrigation Availability:** {irrigation}
    - **Farm Size:** {farm_size} acres
    **Recent Weather Data (Last 30 Days):**
    - **Statistical Summary:**\n{weather_summary}
    - **Recent 15-Day Trend:**\n{recent_weather}
    **--- YOUR TASK ---**
    Create a detailed, two-part report in clear, well-formatted markdown.

    **Part 1: Detailed Climate Analysis**
    - Analyze the provided weather data for {location_name}.
    - Discuss key parameters: Temperature (T2M), Precipitation (PRECTOTCORR), Relative Humidity (RH2M), and Wind Speed (WS2M).
    - Describe the overall climate profile based on this data (e.g., "hot and humid with moderate rainfall").
    - Explain the implications of this climate for agricultural activities in the coming weeks. Mention any potential risks like heat stress, waterlogging, or high winds.

    **Part 2: Personalized Crop Advisory**
    - Based on the climate analysis and the farmer's context, provide a practical crop plan.
    - **Primary Crop Recommendation:** Suggest a suitable crop, a specific variety, the justification for choosing it, and an estimated yield/income.
    - **Alternative Crop Suggestion:** Suggest a secondary option and explain its benefits (e.g., risk diversification, lower water requirement).
    - **Actionable Planting Calendar & Guide:** Provide simple, step-by-step instructions for land preparation, sowing time, seed rate, and water management tailored to the recommended crop.
    - **Sustainable Farming Practices:** Suggest affordable NPK fertilizer applications (including organic alternatives) and Integrated Pest Management (IPM) techniques.
    **Disclaimer:** Start the entire response with a short disclaimer advising the farmer to cross-verify all recommendations with local agricultural extension services.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the report: {e}"

def process_and_display_prediction(image_file):
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
        class_labels = [
            'Apple___Apple_scab', 'Apple___Black_rot', # ... (rest of your class labels)
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

# ====================================================================
#                    STREAMLIT UI LAYOUT
# ====================================================================

st.title("🌱 Smart Agriculture AI Dashboard")
st.markdown("A one-stop AI assistant for **crop planning, plant health, hyperspectral analysis, and weather insights**.")
st.divider()

# --- Initialize session state ---
if 'lat' not in st.session_state: st.session_state['lat'] = None
if 'lon' not in st.session_state: st.session_state['lon'] = None
if 'suggested_soil' not in st.session_state: st.session_state['suggested_soil'] = 0
if 'suggested_irrigation' not in st.session_state: st.session_state['suggested_irrigation'] = 0

# --------------------------------------------------------------------
# 📍 Step 1: Farm Location & Setup
# --------------------------------------------------------------------
st.header("📍 Step 1: Enter Your Farm Details")
st.info("Your location is used to automatically fetch weather data and provide AI-powered suggestions.")

col1, col2 = st.columns([2, 1])
with col1:
    location_name = st.text_input("Farm Location (e.g., 'Pune, Maharashtra')", key="location_name")
    if st.button("Set Location"):
        with st.spinner(f"Finding coordinates for {location_name}..."):
            lat, lon = get_coords_from_place_name(location_name)
            if lat and lon:
                st.session_state['lat'], st.session_state['lon'] = lat, lon
                st.success(f"📍 Location Set: {location_name} (Lat: {lat:.4f}, Lon: {lon:.4f})")
            else:
                st.error(f"Could not find coordinates for '{location_name}'. Please be more specific.")
with col2:
    farm_size = st.number_input("Farm Size (in acres)", min_value=0.5, max_value=500.0, value=2.5, step=0.5)

# --- AI Suggestion for Soil and Irrigation ---
if st.session_state['lat'] and GEMINI_API_KEY:
    if st.button("🤖 Suggest Soil & Irrigation for me"):
        with st.spinner("AI is analyzing your region..."):
            soil, irrigation = get_soil_and_irrigation_suggestion(location_name)
            if soil and irrigation:
                soil_options = ["Alluvial Soil", "Black (Regur) Soil", "Red and Yellow Soil", "Laterite Soil", "Arid (Desert) Soil", "Saline Soil", "Peaty (Marshy) Soil", "Forest and Mountain Soil"]
                irrigation_options = ["Rain-fed (No irrigation)", "Limited (Canal/Well, not regular)", "Good (Drip/Sprinkler/Canal)"]
                try:
                    st.session_state['suggested_soil'] = soil_options.index(soil)
                except ValueError:
                    st.warning(f"AI suggested an unknown soil type: '{soil}'. Please select manually.")
                try:
                    st.session_state['suggested_irrigation'] = irrigation_options.index(irrigation)
                except ValueError:
                    st.warning(f"AI suggested an unknown irrigation type: '{irrigation}'. Please select manually.")
                st.success("AI suggestions have been applied below!")

col3, col4 = st.columns(2)
with col3:
    soil_type = st.selectbox("Soil Type", ["Alluvial Soil", "Black (Regur) Soil", "Red and Yellow Soil", "Laterite Soil", "Arid (Desert) Soil", "Saline Soil", "Peaty (Marshy) Soil", "Forest and Mountain Soil"], index=st.session_state['suggested_soil'])
with col4:
    irrigation = st.selectbox("Irrigation Availability", ["Rain-fed (No irrigation)", "Limited (Canal/Well, not regular)", "Good (Drip/Sprinkler/Canal)"], index=st.session_state['suggested_irrigation'])
st.divider()

# --------------------------------------------------------------------
# 🌾 Step 2: AI Crop & Climate Report
# --------------------------------------------------------------------
st.header("🌾 Step 2: AI Crop & Climate Report")
st.write("Get a unified report with detailed climate analysis and crop suggestions for your farm.")
if st.button("📝 Generate Combined Agri-Report"):
    if not st.session_state['lat'] or not st.session_state['lon']:
        st.warning("Please set your farm location in Step 1 first.")
    elif not GEMINI_API_KEY:
        st.error("Cannot generate a report. The Gemini API key is not configured.")
    else:
        with st.spinner("Fetching weather data and consulting our AI agronomist... This may take a moment."):
            weather_df = fetch_nasa_power_data(st.session_state['lat'], st.session_state['lon'])
            if weather_df is None or weather_df.empty:
                st.error("Failed to fetch weather data for the report. Please check the location and try again.")
            else:
                st.success("Weather data fetched successfully!")
                report = generate_combined_report(st.session_state['lat'], st.session_state['lon'], soil_type, irrigation, farm_size, weather_df, location_name)
                st.subheader("Your Combined Agricultural Report")
                st.markdown(report)
st.divider()

# (The rest of the code for Steps 3, 4, 5, and the Report Section remains the same as your previous version)
# ... (omitted for brevity, please paste the rest of your code here)
# --------------------------------------------------------------------
# 🍃 Step 3: Plant Health Check
# --------------------------------------------------------------------
st.header("🍃 Step 3: Plant Health Check")
st.write("Upload a leaf photo to check for plant diseases and get treatment suggestions.")
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    process_and_display_prediction(uploaded_file)
st.divider()

# --------------------------------------------------------------------
# 🌈 Step 4: Hyperspectral Analysis
# --------------------------------------------------------------------
st.header("🌈 Step 4: Hyperspectral Data Analysis")
st.write("Paste hyperspectral reflectance data to classify crops or land cover using a trained ONNX model.")

if hyperspectral_model:
    default_data = ','.join([f"{np.random.rand()*0.1 + 0.2:.4f}" for _ in range(200)])
    spectral_data_input = st.text_area("Paste Hyperspectral Data (comma separated values)", default_data, height=150)

    if st.button("Run Hyperspectral Analysis"):
        try:
            spectral_values = [float(val.strip()) for val in spectral_data_input.split(',')]
            num_bands = len(spectral_values)
            data_reshaped = np.array(spectral_values).reshape(1, 1, 1, num_bands, 1)
            data_final = data_reshaped.astype(np.float32)

            with st.spinner("Analyzing with ONNX model..."):
                input_name = hyperspectral_model.get_inputs()[0].name
                output_name = hyperspectral_model.get_outputs()[0].name
                result = hyperspectral_model.run([output_name], {input_name: data_final})
                
                prediction_array = result[0]
                predicted_class_index = np.argmax(prediction_array, axis=1)[0]
                
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
                st.success(f"**Predicted Land Cover:** {predicted_label}")
                st.metric("Prediction Confidence", f"{confidence:.2f}%")
            else:
                st.error("Prediction index is out of bounds. Check the model's class labels.")
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
else:
    st.warning("Hyperspectral model not loaded. Please ensure 'trainedIndianPinesCSCNN.onnx' is in the 'models' directory.")
st.divider()

# --------------------------------------------------------------------
# ☀️ Step 5: Weather Insights
# --------------------------------------------------------------------
st.header("☀️ Step 5: Weather Insights")
st.write("Get the latest 30-day weather data from NASA POWER for your farm's location.")
if st.button("Fetch Weather Data"):
    if not st.session_state['lat'] or not st.session_state['lon']:
        st.warning("Please set your farm location in Step 1 first.")
    else:
        with st.spinner("Fetching weather data..."):
            df = fetch_nasa_power_data(st.session_state['lat'], st.session_state['lon'])
            if df is None or df.empty:
                st.error("Failed to fetch NASA POWER data.")
            else:
                st.success("Data fetched successfully!")
                st.subheader(f"Latest 30-Day Weather Data for {location_name}")
                st.dataframe(df)
st.divider()

# --------------------------------------------------------------------
# 📄 Report Section (Placeholder)
# --------------------------------------------------------------------
st.header("📄 Generate Report")
st.write("This feature is coming soon! It will allow you to download a summary of your generated plans and analyses.")
if st.button("Download PDF Report"):
    st.info("PDF report generation functionality is under development.")