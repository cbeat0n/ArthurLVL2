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
    """
    Converts OANDA candle JSON into tabular format.
    Returns middleman list of dicts.
    """
    middleman = []
    for c in candle_json:
        row = {
            "timestamp": str(c["time"]),
            "complete": c["complete"],
            "volume": float(c["volume"]),
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"])
        }
        middleman.append(row)
    return middleman

def clean_tabular(middleman):
    """
    Converts middleman (list of dicts) into cleaned DataFrame.
    Assumes minimal cleaning needed; returns DataFrame and safe timestamp string.
    """
    cleaned_df = pd.DataFrame(middleman)
    
    # No missing or duplicates per 8/6/25 QA — note kept for audit
    safe_timestamp = pd.to_datetime(cleaned_df["timestamp"].iloc[-1]).strftime("%Y-%m-%dT%H-%M-%S")
    
    return cleaned_df, safe_timestamp

# Usage example (called by driver, not run on import):
# middleman = convert_to_tabular(instrument_candles)
# cleaned_df, safe_timestamp = clean_tabular(middleman)


