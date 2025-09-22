### quantitative_extract.py is a modular and agnostic framework that
### communicates server side with data vendors using DataVendor interface and returns 
### candlebody information which is sent to transform_cleaned.py


## External Dependencies
import requests # Library for HTTP requests

## Internal Dependencies
from utils.generic import DataVendor , Platform # Non circular import for generic placeholders


def fetch_candles(mode : str , vendor : DataVendor , platform : Platform , instrument="USD_JPY", intake="S5", count=3600, price="M", timezone="UTC"):
    """
    fetch_candles fetches the candles for the horizon at intake.
    It returns either a JSON with the candlebody or the ValueError.

    Inputs:
    mode : str - mode of trading
    vendor : DataVendor - Vendor for use -- Currently OANDA is both as an inherited Platform
    platform : Platform - Platform for trading -- Currently being used, will be DataVendor later
    instrument : str - financial instrument being traded
    intake : str - intake sample frequency
    count : int - number of samples that defines the horizon, intake * count is integer divisible by
    horizon, defined elsewhere
    price : str - midpoint or bid / ask prices, usually midpoint for forecasting
    timezone : str - timezone of calculations done, different from timezone used in account information pulls.
    """

    url = platform.make_url(mode , "instrument" , 0)

    headers = {"Authorization": f"Bearer {platform.authorization}"}

    params = {"granularity" : intake,"count" : count,"price" : price,"alignmentTimezone" : timezone}

    response = requests.get(url, headers=headers, params=params) # HTTP Request from url , headers , params

    if response.status_code == 200: # Error handling
        return {"candles" : response.json().get("candles" , []) , "instrument" : instrument , "intake" : intake}
    else:
        raise ValueError(f"Error {response.status_code}: {response.text}")
