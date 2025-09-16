# nasa_power.py
# ============================================================
# Fetch NASA POWER environmental data
# ============================================================

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_nasa_power_data(lat: float, lon: float, days: int = 365 * 5) -> pd.DataFrame:
    """
    Fetch daily environmental data from NASA POWER API for a given latitude & longitude.
    Parameters:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        days (int): Number of past days of data to fetch (default=5 years).
    Returns:
        pd.DataFrame: A cleaned dataframe with daily weather parameters, or an empty dataframe on error.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    params = {
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "latitude": lat,
        "longitude": lon,
        "community": "AG",
        "parameters": "T2M,PRECTOTCORR,RH2M,WS2M", # Added Wind Speed (WS2M)
        "format": "JSON",
        "header": "true",
    }

    try:
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        response = requests.get(url, params=params, timeout=60) # Increased timeout for larger data
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()

        if "properties" not in data or "parameter" not in data["properties"]:
            print("⚠️ NASA POWER data is missing expected 'properties' or 'parameter' fields.")
            return pd.DataFrame()

        df = pd.DataFrame(data["properties"]["parameter"])
        df['time'] = pd.to_datetime(df.index, format='%Y%m%d')
        df.set_index('time', inplace=True)

        # Replace -999 with NaN and forward-fill missing values
        df.replace(-999, np.nan, inplace=True)
        df.ffill(inplace=True)

        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching NASA POWER data: {e}")
        return pd.DataFrame()
    except (KeyError, ValueError) as e:
        print(f"❌ Error parsing NASA POWER data: {e}")
        return pd.DataFrame()

def get_yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the yearly average for each weather parameter.
    """
    if df.empty:
        return pd.DataFrame()

    # Resample the data by year and calculate the mean
    yearly_summary = df.resample('Y').mean()
    yearly_summary.index = yearly_summary.index.year
    return yearly_summary
