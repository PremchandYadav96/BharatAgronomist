# BharatAgronomist
A Product of RICE(RevolutionInCultivatingExcellence)

BharatAgronomist is a comprehensive, AI-powered web application designed to assist farmers in making informed decisions for their agricultural practices. It provides a suite of tools for crop planning, plant health monitoring, weather analysis, and more, all accessible through a user-friendly interface.

## Features

*   **Farm Location Setup**: Select your farm's location using an interactive map or voice commands to receive localized data and recommendations.
*   **AI-Powered Suggestions**: Get AI-driven recommendations for soil type and irrigation methods based on your farm's location.
*   **Crop and Climate Reports**: Generate detailed reports that include climate analysis and personalized crop suggestions.
*   **Plant Health Analysis**: Upload images of plant leaves to detect diseases and receive treatment advice.
*   **Live Disease Detection**: Use your device's camera for real-time plant disease identification.
*   **Hyperspectral Analysis**: Analyze hyperspectral data to classify land cover and crop types.
*   **Weather Insights**: Access current and historical weather data from NASA POWER for your specific location.
*   **Fertilizer Recommendations**: Receive fertilizer suggestions tailored to your soil type and selected crops.
*   **Market Price Information**: Fetch the latest market prices for various commodities in India.
*   **Multilingual Support**: The application is available in English, Hindi, and Marathi.

## Directory Structure

```
.
├── app.py                  # Main Streamlit application file
├── requirements.txt        # Python dependencies
├── nasa_power.py           # Module for fetching NASA POWER API data
├── templates/
│   └── index.html          # HTML template for the UI
├── models/                 # Contains the machine learning models
│   ├── plant_disease_model.h5  # Plant disease detection model
│   └── trainedIndianPinesCSCNN.onnx # Hyperspectral analysis model
└── utils/                  # Utility modules
    ├── fertilizer.py       # Provides fertilizer recommendations
    ├── inference.py        # Functions for model inference
    ├── market_data.py      # Fetches market price data
    ├── pdf_report.py       # Generates PDF reports
    ├── preprocess.py       # Data preprocessing functions
    ├── translator.py       # Handles localization and translations
    └── ui_helpers.py       # Helper functions for the Streamlit UI
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/BharatAgronomist.git
    cd BharatAgronomist
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API keys:**
    - Create a `.streamlit/secrets.toml` file.
    - Add your API keys to this file:
      ```toml
      GEMINI_API_KEY = "your_gemini_api_key"
      DATA_GOV_IN_API_KEY = "your_data_gov_in_api_key"
      ASSEMBLYAI_API_KEY = "your_assemblyai_api_key"
      ```

## Usage

To run the application, use the following command:

```bash
streamlit run app.py
```

The application will be accessible in your web browser at `http://localhost:8501`.
