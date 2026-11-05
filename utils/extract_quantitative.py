### quantitative_extract.py is a modular and agnostic framework that
### communicates server side with data vendors using DataVendor interface and returns 
### candlebody information which is sent to transform_cleaned.py


## External Dependencies
import requests # Library for HTTP requests


## Internal Dependencies
from utils.http_helpers import json_or_raise


def fetch_candles(mode : str , vendor , platform, instrument="USD_JPY", intake="S5", count=3600, price="M", timezone="UTC"):
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


    params = {"granularity" : intake,"count" : count,"price" : price,"alignmentTimezone" : timezone}

    response = platform.config.session.get(url = url, params=params)
    
    body = json_or_raise(response, f"candles {instrument} {intake}")
    
    candles = [c for c in body.get("candles", []) if c.get("complete", True)]
    
    return {"candles" : candles , "instrument" : instrument , "intake" : intake}
