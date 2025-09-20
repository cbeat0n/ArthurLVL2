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

# Internal Dependencies
from utils.generic import DataVendor , Platform

# Load environment variables from the .env file
load_dotenv()

# Retrieve OANDA API Key from .env file
oanda_demo_api_key = os.getenv("OANDADEMOAPIKEY")

## FOR LIVE TRADING:
## oanda_api_key = os.getenv("OANDAAPIKEY")



def fetch_candles(mode : str , vendor : DataVendor , platform : Platform , instrument="USD_JPY", intake="S5", count=3600, price="M", timezone="UTC"):
    
    # Technically ETL-Base runs off vendors. However OANDA does not separate these
    # two services so modularly agnostic code will intake a vendor and a platform.

    ### IMPORTANT!! THIS IS BUILT TO RUN ON OANDA BECAUSE THAT IS WHAT IS BEING USED.
    ### COMPLETE MODULAR AGNOSTIC CODE WOULD USE A VENDOR. 
    ### 09182025 1140 CST LOOKING INTO FINAGE ETC AND SEP PLATFORMS


    url = platform.make_url(mode , "instrument" , 0)

    headers = {
        "Authorization": f"Bearer {oanda_demo_api_key}"
    }

    params = {
        "granularity": intake,
        "count": count,
        "price": price,
        "alignmentTimezone": timezone
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json().get("candles", []) , instrument , intake
    else:
        raise ValueError(f"Error {response.status_code}: {response.text}")
