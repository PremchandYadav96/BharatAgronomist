# 🌾 BharatAgronomist
A Product of **[RICE (Revolution In Cultivating Excellence)](https://rice-24.vercel.app/)**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&style=for-the-badge)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Framework-red?logo=streamlit&style=for-the-badge)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&style=for-the-badge)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

![BharatAgronomist Banner](https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExbWhtYjZ4dzJiZm9tN3R6cWlhbWNyZXppYzg3cGxobnpta2I1Z3UxNCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/NatbVZ9NsLwU8/giphy.gif)

---

## 🎯 The Problem
Indian agriculture, the backbone of our economy, faces unprecedented challenges: unpredictable climate patterns, soil degradation, pest infestations, and a significant information gap between labs and land. Farmers often rely on traditional methods that are no longer sufficient to guarantee yield, leading to financial instability and resource wastage.

## ✨ Our Solution
**BharatAgronomist** is a next-generation, AI-powered agricultural intelligence platform designed to bridge this gap. We provide a holistic, data-driven assistant that empowers every farmer with the tools of modern precision agriculture, right at their fingertips. Our mission is to make farming more **profitable, sustainable, and resilient.**

---

## 🎥 Live Demo
*(A short video demonstrating the key features of the application)*
**[Watch the Demo on YouTube](https://your-youtube-link-here.com)**

---

## 🚀 Core Features

### 🧠 Decision Intelligence
*   **📍 Geospatial Farm Setup:** Automatically fetch farm coordinates using location names.
*   **🤖 AI-Powered Soil & Irrigation Analysis:** Get intelligent suggestions for soil type and irrigation methods based on regional data.
*   **📊 Combined Agri-Report:** Generate a comprehensive report with detailed climate analysis and a personalized crop advisory in one click.
*   **📈 Live Market Insights:** Access real-time commodity prices from local mandis via the Data.gov.in API to sell produce at the right time.
*   **🧪 Smart Fertilizer Recommendations:** Receive custom NPK and micronutrient advice based on soil, crop, and location.

### 🌿 Crop Health & Monitoring
*   **🌱 Image-Based Disease Detection:** Upload a leaf image to instantly diagnose diseases with 98% accuracy.
*   **📷 Live Camera Diagnostics:** Use your device's camera for real-time crop health scanning in the field.
*   **🌈 Hyperspectral Land Cover Analysis:** Analyze hyperspectral data using a pre-trained ONNX model to classify crop types and assess land health.

### ☀️ Accessibility & Environment
*   **🗣️ Multilingual Voice Commands:** Interact with the app in **English, Hindi, and Marathi** using advanced voice recognition.
*   **☀️ Hyperlocal Weather Insights:** Get a 30-day forecast and historical weather data from the NASA POWER API, tailored to your farm's exact coordinates.
*   **📄 PDF Report Generation:** Download all AI-generated advisories and analyses as a portable PDF report.

---

## 🧠 Tech Stack & Architecture
Our platform is built on a modular, service-oriented architecture to ensure scalability and maintainability.

| Category                | Technology                               | Purpose                                                 |
| ----------------------- | ---------------------------------------- | ------------------------------------------------------- |
| **Frontend**            | Streamlit                                | Interactive & data-centric user interface               |
| **AI & Machine Learning** | TensorFlow (Keras), ONNX Runtime         | Plant Disease Detection & Hyperspectral Analysis        |
| **Generative AI**       | Google Gemini API                        | AI Agronomist, Soil/Irrigation Suggestions, Report Gen  |
| **Geospatial**          | Geopy                                    | Geocoding farm locations                                |
| **Data APIs**           | NASA POWER, Data.gov.in                  | Weather Insights & Live Market Prices                   |
| **Voice Recognition**   | AssemblyAI                               | Multilingual voice commands                             |
| **Deployment**          | Streamlit Community Cloud (or similar)   | Cloud-based hosting                                     |

---

## 🏗️ Project Structure
The repository is structured to separate concerns, making the codebase clean and easy to navigate.

```
BharatAgronomist/
├── app.py                  # Main Streamlit application logic
├── requirements.txt        # List of Python dependencies for pip
├── .streamlit/
│   └── secrets.toml        # Securely store API keys (NEVER commit this)
├── models/
│   ├── plant_disease_model.h5 # Trained Keras model for disease detection
│   └── trainedIndianPinesCSCNN.onnx # ONNX model for hyperspectral analysis
├── nasa_power.py           # Helper module for NASA POWER API integration
└── utils/                  # Utility scripts for modular functions
    ├── inference.py        # Functions for model prediction
    ├── market_data.py      # Module to fetch market price data
    ├── pdf_report.py       # Functions for generating PDF reports
    └── translator.py       # Language translation and localization
```

---

## ⚙️ Quick Start Guide

**1. Clone the Repository:**
```bash
git clone https://github.com/your-username/BharatAgronomist.git
cd BharatAgronomist
```

**2. Create & Activate Virtual Environment:**
```bash
# It is highly recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Configure API Keys:**
Create a file `secrets.toml` inside a `.streamlit` folder.
```
.
├── .streamlit/
│   └── secrets.toml
└── app.py
```
Add your API keys to `secrets.toml`:
```toml
GEMINI_API_KEY = "your_gemini_api_key"
DATA_GOV_IN_API_KEY = "your_data_gov_in_api_key"
ASSEMBLYAI_API_KEY = "your_assemblyai_api_key"
```

**5. Run the Application:**
```bash
streamlit run app.py
```
Open your browser and go to `http://localhost:8501`.

---

## 🏆 Meet The Team: RICE
We are RICE (Revolution In Cultivating Excellence), a team of passionate innovators dedicated to solving real-world agricultural challenges through technology.

| Name                | Role                |
| ------------------- | ------------------- |
| [Your Name]         | Team Lead / AI Dev  |
| [Teammate's Name]   | Full-Stack Dev      |
| [Teammate's Name]   | UI/UX Designer      |
| [Teammate's Name]   | Data Scientist      |

---

## 🛣️ Future Roadmap
*   **Drone Integration:** Process drone imagery for large-scale crop stress analysis.
*   **Supply Chain Linkage:** Connect farmers directly with buyers through the platform.
*   **Water Management Module:** Provide advisories on efficient water usage based on soil moisture data.
*   **Offline Capability:** Develop a lightweight, offline-first version for areas with poor connectivity.

---
Crafted with ❤️ for the farmers of India.

*Helping farmers → Save Time, Increase Yield, and Cultivate Excellence!* 🌾✨
