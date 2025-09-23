### transform_cleaned.py is a modular and agnostic framework that performs
### proprietary ETL as a part of ETL-Base for Arthur.

## External Dependencies
import pandas as pd # Library for data manipulation
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

    ## Maps are made for OANDA.
    horizon_rule = {"M1" : "1T", "M2" : "2T", "M4" : "4T", "M5" : "5T", "M10" : "10T"}[horizon]

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




