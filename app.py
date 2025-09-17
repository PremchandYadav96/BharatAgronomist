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
import pandas as pd
import tensorflow as tf
from PIL import Image
import google.generativeai as genai
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import onnxruntime as ort
from nasa_power import fetch_nasa_power_data # Ensure nasa_power.py is in the same directory
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from streamlit_folium import st_folium
import folium
from nasa_power import get_yearly_summary, get_monthly_summary
from utils.translator import t
from utils.market_data import fetch_market_prices
from utils.fertilizer import get_fertilizer_recommendation

# ====================================================================
# Page Configuration & API Setup
# ====================================================================
st.set_page_config(
    page_title="Smart Agriculture AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Language Selection")
lang = st.sidebar.selectbox("Select Language", ["en", "hi", "mr", "te", "pa"], format_func=lambda x: {"en": "English", "hi": "हिंदी", "mr": "मराठी", "te": "తెలుగు", "pa": "ਪੰਜਾਬੀ"}[x])
if 'lang' not in st.session_state or st.session_state['lang'] != lang:
    st.session_state['lang'] = lang
    st.rerun()

# ####################################################################
# # --- Secure API Key Handling with Streamlit Secrets ---
# ####################################################################
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except (KeyError, Exception) as e:
    st.warning("⚠️ Gemini API key not found or invalid. AI features will be disabled.")
    GEMINI_API_KEY = None
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

plant_model = load_plant_model()

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

def generate_combined_report(lat, lon, soil_type, irrigation, farm_size, weather_df, location_name, language='en'):
    """Generates a combined climate and crop report using Gemini AI and translates if needed."""
    if not GEMINI_API_KEY: return "Cannot generate report: Gemini API key is not configured."

    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    weather_summary = weather_df.describe().to_string()
    recent_weather = weather_df.tail(15).to_string()

    # Step 1: Generate the report in English for consistency
    english_prompt = f"""
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
    Create a detailed, two-part report in clear, well-formatted markdown. Respond in English.

    **Part 1: Detailed Climate Analysis**
    - Analyze the provided weather data for {location_name}.
    - Discuss key parameters: Temperature (T2M), Precipitation (PRECTOTCORR), Relative Humidity (RH2M), and Wind Speed (WS2M).
    - Describe the overall climate profile based on this data.
    - Explain the implications of this climate for agricultural activities.

    **Part 2: Personalized Crop Advisory**
    - Based on the climate analysis and farmer's context, provide a practical crop plan.
    - **Primary Crop Recommendation:** Suggest a suitable crop, a specific variety, and justification.
    - **Alternative Crop Suggestion:** Suggest a secondary option and its benefits.
    - **Actionable Planting Calendar & Guide:** Provide simple, step-by-step instructions.
    - **Sustainable Farming Practices:** Suggest NPK fertilizers and IPM techniques.
    **Disclaimer:** Start the entire response with a short disclaimer advising the farmer to cross-verify all recommendations with local agricultural extension services.
    """
    try:
        english_response = model.generate_content(english_prompt)
        english_report = english_response.text
    except Exception as e:
        return f"An error occurred while generating the report: {e}"

    # Step 2: Translate the report if a different language is selected
    if language != 'en':
        lang_map = {"hi": "Hindi", "mr": "Marathi", "te": "Telugu", "pa": "Punjabi"}
        target_language = lang_map.get(language, "English")

        translation_prompt = f"Translate the following agricultural report into {target_language}. Preserve the markdown formatting, including headings, bold text, and bullet points.\n\nReport:\n{english_report}"

        try:
            translation_response = model.generate_content(translation_prompt)
            return translation_response.text
        except Exception as e:
            st.warning(f"Could not translate the report to {target_language}. Displaying the English version instead.")
            # Fallback to English report if translation fails
            return english_report

    return english_report

plant_disease_class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

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
        predicted_class_label = plant_disease_class_labels[predicted_class_index]
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

st.title(f"🌱 {t('app_title')}")
st.markdown(t('app_tagline'))
st.divider()

# --- Initialize session state ---
if 'lat' not in st.session_state: st.session_state['lat'] = None
if 'lon' not in st.session_state: st.session_state['lon'] = None
if 'suggested_soil' not in st.session_state: st.session_state['suggested_soil'] = 0
if 'suggested_irrigation' not in st.session_state: st.session_state['suggested_irrigation'] = 0
if 'report_content' not in st.session_state: st.session_state['report_content'] = ""

# --------------------------------------------------------------------
# 📍 Step 1: Enter Your Farm Details
# --------------------------------------------------------------------
st.markdown("""
    <div style="background-color: #2E8B57; padding: 10px; border-radius: 5px;">
        <h2 style="color: white; text-align: center;">📍 Step 1: Enter Your Farm Details</h2>
    </div>
    """, unsafe_allow_html=True)
st.info("Your location is used to automatically fetch weather data and provide AI-powered suggestions.")

col1, col2 = st.columns([2, 1])
with col1:
    st.write("Click on the map to select your farm's location.")
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    map_data = st_folium(m, width=700, height=500)

    if map_data and map_data["last_clicked"]:
        lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        st.session_state['lat'], st.session_state['lon'] = lat, lon
        st.success(f"📍 Location Set: (Lat: {lat:.4f}, Lon: {lon:.4f})")

        # --- Auto-suggest soil and irrigation ---
        geolocator = Nominatim(user_agent="smart_agri_dashboard")
        try:
            location = geolocator.reverse((lat, lon), exactly_one=True)
            location_name = location.address if location else f"{lat:.4f}, {lon:.4f}"
            st.session_state['location_name'] = location_name

            with st.spinner("AI is analyzing your region for soil and irrigation types..."):
                soil, irrigation = get_soil_and_irrigation_suggestion(location_name)
                if soil and irrigation:
                    soil_options = ["Alluvial Soil", "Black (Regur) Soil", "Red and Yellow Soil", "Laterite Soil", "Arid (Desert) Soil", "Saline Soil", "Peaty (Marshy) Soil)", "Forest and Mountain Soil"]
                    irrigation_options = ["Rain-fed (No irrigation)", "Limited (Canal/Well, not regular)", "Good (Drip/Sprinkler/Canal)"]
                    try:
                        st.session_state['suggested_soil'] = soil_options.index(soil)
                    except ValueError:
                        st.warning(f"AI suggested an unknown soil type: '{soil}'. Please select manually.")
                    try:
                        st.session_state['suggested_irrigation'] = irrigation_options.index(irrigation)
                    except ValueError:
                        st.warning(f"AI suggested an unknown irrigation type: '{irrigation}'. Please select manually.")
                    st.success("✅ AI suggestions for soil and irrigation have been auto-filled!")
        except (GeocoderTimedOut, GeocoderUnavailable):
            st.error("Could not connect to geocoding service to get location name.")
            st.session_state['location_name'] = f"{lat:.4f}, {lon:.4f}"

with col2:
    farm_size = st.number_input("Farm Size (in acres)", min_value=0.5, max_value=500.0, value=2.5, step=0.5)
    location_name_display = st.text_input("Location Name", st.session_state.get('location_name', ''), help="Automatically fetched from map coordinates.")

col3, col4 = st.columns(2)
with col3:
    soil_type_options = ["Alluvial Soil", "Black (Regur) Soil", "Red and Yellow Soil", "Laterite Soil", "Arid (Desert) Soil", "Saline Soil", "Peaty (Marshy) Soil", "Forest and Mountain Soil"]
    soil_type = st.selectbox("Soil Type", soil_type_options, index=st.session_state.get('suggested_soil', 0))
    st.session_state['soil_type'] = soil_type
with col4:
    irrigation_options = ["Rain-fed (No irrigation)", "Limited (Canal/Well, not regular)", "Good (Drip/Sprinkler/Canal)"]
    irrigation = st.selectbox("Irrigation Availability", irrigation_options, index=st.session_state.get('suggested_irrigation', 0))
st.divider()

# --------------------------------------------------------------------
# 🌾 Step 2: AI Crop & Climate Report
# --------------------------------------------------------------------
st.markdown("""
    <div style="background-color: #2E8B57; padding: 10px; border-radius: 5px;">
        <h2 style="color: white; text-align: center;">🌾 Step 2: AI Crop & Climate Report</h2>
    </div>
    """, unsafe_allow_html=True)
st.info("Get a unified report with detailed climate analysis and crop suggestions for your farm.")
if st.button("📝 Generate Combined Agri-Report"):
    if not st.session_state['lat'] or not st.session_state['lon']:
        st.warning("Please set your farm location in Step 1 first.")
    elif not GEMINI_API_KEY:
        st.error("Cannot generate a report. The Gemini API key is not configured.")
    else:
        with st.spinner("Fetching weather data and consulting our AI agronomist... This may take a moment."):
            # Use the location_name from the session state which is now reverse-geocoded
            location_name_for_report = st.session_state.get('location_name', f"{st.session_state['lat']:.4f}, {st.session_state['lon']:.4f}")
            weather_df = fetch_nasa_power_data(st.session_state['lat'], st.session_state['lon'])
            if weather_df is None or weather_df.empty:
                st.error("Failed to fetch weather data for the report. Please check the location and try again.")
            else:
                st.success("Weather data fetched successfully!")
                report = generate_combined_report(st.session_state['lat'], st.session_state['lon'], st.session_state['soil_type'], irrigation, farm_size, weather_df, location_name_for_report, st.session_state.get('lang', 'en'))
                st.session_state['report_content'] = report
                st.subheader("Your Combined Agricultural Report")
                st.markdown(report)
st.divider()

# --------------------------------------------------------------------
# 🍃 Step 3: Plant Health Check (Image Upload)
# --------------------------------------------------------------------
st.markdown("""
    <div style="background-color: #2E8B57; padding: 10px; border-radius: 5px;">
        <h2 style="color: white; text-align: center;">🍃 Step 3: Plant Health Check (Image Upload)</h2>
    </div>
    """, unsafe_allow_html=True)
st.info("Upload an image of a plant leaf to detect diseases.")
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    process_and_display_prediction(uploaded_file)
st.divider()

# --------------------------------------------------------------------
# 📸 Step 4: Live Plant Health Check (Webcam)
# --------------------------------------------------------------------
st.markdown("""
    <div style="background-color: #2E8B57; padding: 10px; border-radius: 5px;">
        <h2 style="color: white; text-align: center;">📸 Step 4: Live Plant Health Check (Webcam)</h2>
    </div>
    """, unsafe_allow_html=True)
st.info("Use your camera to detect plant diseases in real-time.")

class VideoTransformer(VideoTransformerBase):
    def __init__(self, plant_model, class_labels):
        self.plant_model = plant_model
        self.class_labels = class_labels

    def transform(self, frame):
        img = Image.fromarray(frame.to_ndarray(format="bgr24"))
        img_resized = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        if self.plant_model is not None:
            prediction = self.plant_model.predict(img_array)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction) * 100
            predicted_class_label = self.class_labels[predicted_class_index]

            import cv2
            text = f"{predicted_class_label.replace('___', ' ').replace('_', ' ')} ({confidence:.1f}%)"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            frame_bgr = frame.to_ndarray(format="bgr24")
            cv2.rectangle(frame_bgr, (10, 30 - h - 5), (10 + w, 30 + 5), (0,0,0), -1)
            cv2.putText(frame_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return frame_bgr
        return frame.to_ndarray(format="bgr24")

webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=lambda: VideoTransformer(plant_model, plant_disease_class_labels),
    media_stream_constraints={"video": True, "audio": False},
)
st.divider()

# --------------------------------------------------------------------
# ☀️ Step 5: Weather Insights & Forecast
# --------------------------------------------------------------------
st.markdown("""
    <div style="background-color: #2E8B57; padding: 10px; border-radius: 5px;">
        <h2 style="color: white; text-align: center;">☀️ Step 5: Weather Insights & Forecast</h2>
    </div>
    """, unsafe_allow_html=True)
st.info("Get current and historical weather data for your location.")
if st.button("📊 Fetch Weather Analysis"):
    if not st.session_state['lat'] or not st.session_state['lon']:
        st.warning("Please set your farm location in Step 1 first.")
    else:
        with st.spinner("Fetching current and historical weather data..."):
            df = fetch_nasa_power_data(st.session_state['lat'], st.session_state['lon'])
            if df is None or df.empty:
                st.error("Failed to fetch NASA POWER data.")
            else:
                st.success("Data fetched successfully!")
                location_name_for_weather = st.session_state.get('location_name', 'your selected location')
                st.subheader(f"Latest 30-Day Weather Data for {location_name_for_weather}")
                st.dataframe(df.head())

                st.subheader("Historical Weather Trends (5-Year Averages)")
                yearly_summary = get_yearly_summary(df)
                st.bar_chart(yearly_summary)

                st.subheader("Fungal Disease Risk Alert")
                last_7_days = df.tail(7)
                if not last_7_days.empty and last_7_days['RH2M'].mean() > 75:
                    st.warning("⚠️ High risk of fungal disease! Average humidity has been high recently.")
                else:
                    st.info("No immediate high-risk weather patterns for fungal diseases detected.")
st.divider()

# --------------------------------------------------------------------
# 💰 Step 6: Market Prices
# --------------------------------------------------------------------
st.markdown("""
    <div style="background-color: #2E8B57; padding: 10px; border-radius: 5px;">
        <h2 style="color: white; text-align: center;">💰 Step 6: Market Prices</h2>
    </div>
    """, unsafe_allow_html=True)
st.info("Get the latest market prices for your crops from various markets in India.")

commodity = st.text_input("Enter a crop name (e.g., 'Wheat', 'Paddy')", key="market_crop")
if st.button("📈 Fetch Market Prices"):
    if commodity:
        with st.spinner(f"Fetching market prices for {commodity}..."):
            prices = fetch_market_prices(commodity)
            if isinstance(prices, list) and prices:
                st.success(f"Latest market prices for {commodity}:")
                df_prices = pd.DataFrame(prices)

                # Ensure modal_price is numeric
                df_prices['modal_price'] = pd.to_numeric(df_prices['modal_price'], errors='coerce')
                df_prices.dropna(subset=['modal_price'], inplace=True)

                st.dataframe(df_prices)

                st.subheader("Price Comparison Across Markets")
                chart_data = df_prices.set_index('market')[['modal_price']]
                st.bar_chart(chart_data)
            else:
                st.error(f"Could not fetch market prices for '{commodity}'. Please check the name or try another crop.")
    else:
        st.warning("Please enter a crop name.")
st.divider()

# --------------------------------------------------------------------
# 🌿 Step 7: Fertilizer Recommendation
# --------------------------------------------------------------------
st.markdown("""
    <div style="background-color: #2E8B57; padding: 10px; border-radius: 5px;">
        <h2 style="color: white; text-align: center;">🌿 Step 7: Fertilizer Recommendation</h2>
    </div>
    """, unsafe_allow_html=True)
st.info("Get fertilizer recommendations based on your soil type and crop.")

crop_list = [
    "Apple", "Banana", "Barley", "Brinjal", "Cashew", "Coconut", "Coffee", "Cotton",
    "Gram", "Groundnut", "Guar", "Jowar", "Maize", "Mango", "Millet", "Moong",
    "Mustard", "Okra", "Onion", "Orange", "Paddy", "Potato", "Ragi", "Rubber",
    "Soybean", "Sugarcane", "Sunflower", "Tea", "Tomato", "Tur (Arhar)", "Urad", "Wheat"
]
selected_crop_fertilizer = st.selectbox("Select your crop", crop_list, key="fertilizer_crop")

if st.button("💡 Get Fertilizer Recommendation"):
    if 'soil_type' in st.session_state:
        soil_type = st.session_state['soil_type']
        recommendation = get_fertilizer_recommendation(soil_type, selected_crop_fertilizer)
        st.info(f"**Recommendation for {selected_crop_fertilizer} in {soil_type}:**\n\n{recommendation}")
    else:
        st.warning("Please select your soil type in Step 1 first.")
