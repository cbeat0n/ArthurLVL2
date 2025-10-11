### transform_cleaned.py is a modular and agnostic framework that performs
### proprietary ETL as a part of ETL-Base for Arthur.

## External Dependencies
import pandas as pd # Library for data manipulation
import re # Library for RegEx
from datetime import datetime # Library for time objects

def convert_to_tabular(candlebody : dict):
    """
    convert_to_tabular takes the JSON from fetch_candles and converts it to tabular form.
    It returns a JSON.

    Inputs:
    candlebody: list - the list of dictionaries incoming from fetch_candles
    """
    
    candles = candlebody["candles"]
    tabular = []

    for candle in candles:  
        row = {
            "timestamp": str(candle["time"]),
            "open": float(candle["mid"]["o"]),
            "high": float(candle["mid"]["h"]),
            "low": float(candle["mid"]["l"]),
            "close": float(candle["mid"]["c"]),
            "volume": int(candle["volume"])
        }
        tabular.append(row)
    
    return tabular

def clean_tabular(tabular : list ,  horizon : str):
    """
    clean_tabular takes the tabular JSON from convert_to_tabular and uses the horizon sample frequency
    to make a Pandas Dataframe that has candle information at horizon frequency, proprietarily constructed.
    It returns a JSON.

    Inputs:
    tabular : list - tabular JSON from convert_to_tabular
    horizon : str - horizon sampling frequency for decision rules over zone , to be defined later
    """
    
    cleaned_df = pd.DataFrame(tabular)

    cleaned_df["timestamp"] = pd.to_datetime(cleaned_df["timestamp"] , utc = True , errors = 'coerce')
    cleaned_df = cleaned_df.sort_values('timestamp').set_index('timestamp')

    ## Take Horizon of form MXQN e.g M20Q1 for M20.25 and split it via RegEx
    parts = re.match(r"([A-Z]+\d+)([A-Z]+\d+)", horizon)
    if parts:
        M, Q = parts.groups()
        

    ## Number of seconds for horizon is calculated using modulo arithmetic.
    ## For Sk and MXQN , seconds is ( 15 / k ) * (4 X + N) or 60 seconds per minute and 15 seconds per
    ## quarter minute , hardcoded for S5. There are 3 df because intake, horizon, and zone, but the 
    ## intake is a hardcoded value for division clarity.
    horizon_rule = pd.Timedelta((15 / 5) * (4 * int(M[1:]) + int(Q[1:])) , "s")

    ## Aggregation smooths out each pricing definition by its accorded aggregation
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}

    ## Resamples the cleaned dataframe to horizon sample frequency / decision frequency.
    ## Uses right label and closed values, aggregation function on line 41.
    horizon_df = (
        cleaned_df.resample(horizon_rule, label='right', closed='right').agg(agg)
          .dropna(how = 'any').reset_index())
    
    ## Timestamp uses Zulu sub seconds. Useful post MVP.
    safe_timestamp = pd.to_datetime(horizon_df["timestamp"].iloc[-1]).strftime("%Y-%m-%dT%H-%M-%S")
    
    return {"cleaned" : horizon_df, "safe_ts" : safe_timestamp}




