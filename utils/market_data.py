import requests
import streamlit as st

def fetch_market_prices(commodity: str):
    """
    Fetches market prices for a given commodity from the data.gov.in API.
    """
    api_key = st.secrets.get("DATA_GOV_IN_API_KEY")
    if not api_key:
        return "API key for data.gov.in not found."

    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    params = {
        "api-key": api_key,
        "format": "json",
        "offset": 0,
        "limit": 10,
        "filters[commodity]": commodity,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "records" in data and data["records"]:
            return data["records"]
        else:
            return f"No market price data found for '{commodity}'."
    except requests.exceptions.RequestException as e:
        return f"Error fetching market price data: {e}"
    except (KeyError, ValueError) as e:
        return f"Error parsing market price data: {e}"
