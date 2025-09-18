### transform_cleaned.py is an an agnostic framework that serves to intake
### data (quantitative and/or qualitative) and turn it into a tabular form. 
### This data is not signal data, but it is convertable. In this module, anything
### resembling soup will be feature extracted and then missing, duplicates via 
### timestamps , and possibly outliers will be handled accordingly.

# Dependencies
import pandas as pd # For tabular conversion
from datetime import datetime # For timestamp conversions

# Currently system nominal will need to be updated for more robust DE,SE,Cloud, Cybersecurity, AI/ML needs.
# 08212025 1429 CST


def convert_to_tabular(candle_json):
    
    middleman = []
    for c in candle_json:
        row = {
            "timestamp": str(c["time"]),
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
            "volume": float(c["volume"])
        }
        middleman.append(row)
    return middleman

def clean_tabular(middleman ,  horizon : str):
    
    cleaned_df = pd.DataFrame(middleman)

    cleaned_df["timestamp"] = pd.to_datetime(cleaned_df["timestamp"] , utc = True , errors = 'coerce')
    cleaned_df = cleaned_df.sort_values('timestamp').set_index('timestamp')

    # Maps are made for OANDA, might need to be changed to be completely modular and agnostic.

    horizon_rule = {
        "M1" : "1T",
        "M2" : "2T",
        "M4" : "4T",
        "M5" : "5T",
        "M10" : "10T"
    }[horizon]

    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}

    horizon_df = (
        cleaned_df.resample(horizon_rule, label='right', closed='right').agg(agg)
          .dropna(how = 'any').reset_index()
    )
    
    # No missing or duplicates per 8/6/25 QA — note kept for audit
    safe_timestamp = pd.to_datetime(horizon_df["timestamp"].iloc[-1]).strftime("%Y-%m-%dT%H-%M-%S")
    
    return horizon_df, safe_timestamp




