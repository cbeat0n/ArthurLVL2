### quantitative_extract.py is an agnostic framework that serves to acquire 
### the input stream of financial information used for fastBot. There is no 
### one set technology that is used, and the principles of API input are 
### kept detached so as to allow the development cycle to be flexible.

# Beginning of this file will use functions for LVL2 pruning onward
# for cybersecurity concerns regarding API Keys and for stronger DevOps.


# Dependencies
from dotenv import load_dotenv # Env hides API Key
import os # Works alongside dotenv for getenv function
import json # Keeping for JSON convertability, might go tabular / columnar
import requests # Library for HTTP GET et al requests

# Currently system nominal will need to be updated for more robust DE,SE,Cloud, Cybersecurity, AI/ML needs.
# 08212025 1429 CST

# Load environment variables from the .env file
load_dotenv()

# Retrieve OANDA API Key from .env file
oanda_demo_api_key = os.getenv("OANDADEMOAPIKEY")

## FOR LIVE TRADING LVL1 TEST:
## oanda_api_key = os.getenv("OANDAAPIKEY")

# Modular Extraction v0

# Create base URL.
demo_base_url = "https://api-fxpractice.oanda.com"

## FOR LIVE TRADING LVL1 TEST:
## live_base_url = "https://api-fxtrade.oanda.com"


def fetch_candles(instrument="USD_JPY", granularity="M1", count=120, price="M", timezone="UTC"):
    """
    Fetch candle data from OANDA API.
    
    Args:
        instrument (str): e.g., "USD_JPY"
        granularity (str): e.g., "S5" , "S10
        count (int): number of candles
        price (str): "M" for midpoint, "B" for bid, "A" for ask
        timezone (str): time alignment, usually "UTC"
    
    Returns:
        list[dict]: candle data
    """
    instrument_url = f"/v3/instruments/{instrument}/candles"
    url = demo_base_url + instrument_url

    headers = {
        "Authorization": f"Bearer {oanda_demo_api_key}"
    }

    params = {
        "granularity": granularity,
        "count": count,
        "price": price,
        "alignmentTimezone": timezone
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json().get("candles", []) , instrument , granularity
    else:
        raise ValueError(f"Error {response.status_code}: {response.text}")
