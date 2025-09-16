# nasa_power.py
# ============================================================
# Fetch NASA POWER environmental data & analyze with Gemini AI
# ============================================================

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import google.generativeai as genai


# ===============================
# CONFIG: Gemini API
# ===============================
GEMINI_API_KEY = "AIzaSyCzghG5GiSTX6MnzaxeGXlEuh6aFCec37A"  # replace with your key
genai.configure(api_key=GEMINI_API_KEY)


# ===============================
# Fetch NASA POWER Data
# ===============================
def fetch_nasa_power_data(lat: float, lon: float, days: int = 365) -> pd.DataFrame:
    """
    Fetch daily environmental data from NASA POWER API for given latitude & longitude.
    Parameters:
        lat (float): Latitude of location
        lon (float): Longitude of location
        days (int): Number of past days to fetch (default=365)
    Returns:
        pd.DataFrame: Cleaned dataframe with daily weather parameters
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    params = {
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "latitude": lat,
        "longitude": lon,
        "community": "AG",
        "parameters": "T2M,PRECTOTCORR,RH2M,GWETROOT",
        "format": "JSON",
        "header": "true",
    }

    try:
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "properties" not in data or "parameter" not in data["properties"]:
            return pd.DataFrame()

        df = pd.DataFrame(data["properties"]["parameter"])
        df["time"] = pd.to_datetime(df.index, format="%Y%m%d")
        df.set_index("time", inplace=True)
        df.replace(-999, np.nan, inplace=True)
        df.ffill(inplace=True)

        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching NASA POWER data: {e}")
        return pd.DataFrame()


# ===============================
# Analyze Data with Gemini
# ===============================
def analyze_with_gemini(df: pd.DataFrame, lat: float, lon: float, days: int = 30) -> str:
    """
    Send recent environmental data to Gemini for analysis.
    Parameters:
        df (pd.DataFrame): NASA POWER data
        lat (float): Latitude
        lon (float): Longitude
        days (int): Recent number of days to analyze
    Returns:
        str: AI-generated insights
    """
    if df.empty:
        return "No NASA POWER data available for analysis."

    recent_df = df.tail(days)
    summary_text = recent_df.describe().to_string()

    prompt = f"""
    I have NASA POWER environmental data for the past {days} days 
    for the region at latitude {lat}, longitude {lon}.

    The parameters are:
    - T2M: Average Temperature (°C)
    - PRECTOTCORR: Precipitation (mm/day)
    - RH2M: Relative Humidity (%)
    - GWETROOT: Root Zone Soil Wetness (fraction)

    Here is the statistical summary of the data:
    {summary_text}

    Please provide a detailed analysis including:
    1. General climate/weather conditions for this region.
    2. Suggested crops suitable for cultivation under these conditions.
    3. Potential agricultural risks or anomalies to monitor.
    4. Insights that could help predict plant diseases.
    5. Recommendations for optimizing crop health and yield.
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error analyzing with Gemini: {e}"


# ===============================
# Plot Data (Optional)
# ===============================
def plot_environmental_trends(df: pd.DataFrame, days: int = 30):
    """
    Plot last {days} days of environmental data.
    """
    if df.empty:
        print("No data to plot.")
        return

    df[["T2M", "RH2M", "GWETROOT"]].tail(days).plot(
        figsize=(12, 6), linewidth=2, grid=True,
        title=f"Last {days} Days of NASA POWER Data"
    )
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.show()


# ===============================
# Example Run (for testing only)
# ===============================
if __name__ == "__main__":
    # Example coordinates (Andhra Pradesh)
    sample_lat, sample_lon = 13.19908, 78.74694

    print("🌍 Fetching NASA POWER data...")
    weather_df = fetch_nasa_power_data(sample_lat, sample_lon)

    if not weather_df.empty:
        print("✅ NASA POWER data fetched successfully!")
        print(weather_df.tail())

        print("\n📈 Plotting last 30 days trends...")
        plot_environmental_trends(weather_df, days=30)

        print("\n🤖 Gemini Analysis:")
        insights = analyze_with_gemini(weather_df, sample_lat, sample_lon, days=30)
        print(insights)
    else:
        print("❌ Failed to fetch NASA POWER data.")
