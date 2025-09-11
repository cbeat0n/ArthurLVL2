### historicaldataverification.py is a small framework that allows for the DDT model to be
### tested in a deprecatable manner separate from Arthur's framework in order to
### allow for demo and then live testing without wasting resources. This is akin to 
### finding the y for a ML framework and looking for training accuracy, before demo test
### and live test accuracy. Last but not least something similar to this will be integrated 
### into load_signal or similar so that the predictive accuracy of Arthur can be upgraded
### in real time.

# External
import pandas as pd
import time
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta
import sys
import os
from sklearn.metrics import confusion_matrix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Internal modules
# from utils.quantitative_extract import fetch_candles  # you'll define this
# fetch_historical_candles() is separate from E1 before we fully develop Arthur
# and fetch_historical_candles is only in this file.
from utils.transform_cleaned import convert_to_tabular, clean_tabular
from utils.transform_signal import transform_to_signal, decisionrule
from utils.load_signal import load_signal_matrix

# 1. Historical Date Range
START_DATE = "2025-09-01"
END_DATE = "2025-09-05"
INSTRUMENT = "USD_JPY"
GRANULARITY = "S5"  # S collapses to naive holding... too fast? M10 good, M15 less good, now M30.
# Intuition says B, S will be 20-25% each and H is "too much noise don't trade".
# For MFT, M30 is too much, H clearly too much. M10 and M15 so far.
TRUTH_LOOKAHEAD = 1  # how many future candles to look ahead for labeling Action


def fetch_historical_candles(instrument="USD_JPY", granularity="S5", price="M", date=None, timezone="UTC"):
    """
    Fetch a full day's worth of historical candle data from OANDA using rolling time windows.

    Returns:
        tuple: (combined_candles: list, instrument: str, granularity: str)
    """
    if not date:
        raise ValueError("Must provide a `date` in YYYY-MM-DD format.")

    load_dotenv()
    oanda_demo_api_key = os.getenv("OANDADEMOAPIKEY")
    demo_base_url = "https://api-fxpractice.oanda.com"
    instrument_url = f"/v3/instruments/{instrument}/candles"
    url = demo_base_url + instrument_url

    headers = {
        "Authorization": f"Bearer {oanda_demo_api_key}"
    }

    # Convert date string to datetime object
    start_time = datetime.strptime(date, "%Y-%m-%d")
    end_time = start_time + timedelta(days=1)

    # Define safe window chunk size based on granularity
    # For S5, safe limit is 4 hours
    chunk_size = timedelta(hours=4)

    all_candles = []
    current_time = start_time

    while current_time < end_time:
        chunk_start = current_time
        chunk_end = min(current_time + chunk_size, end_time)

        params = {
            "from": chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "granularity": granularity,
            "price": price,
            "alignmentTimezone": timezone
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            candles = response.json().get("candles", [])
            all_candles.extend(candles)
        else:
            raise ValueError(f"OANDA API error {response.status_code}: {response.text}")

        current_time += chunk_size

    return all_candles, instrument, granularity

def pip_size(symbol = "USD_JPY"):
    """
    Defines the size of a pip for an FX pair.
    """

    if symbol == "USD_JPY":
        return 0.01
    else:
        return 0.0001 # Generic pip and can be made for my pair options.

def generate_true_actions(cleaned_df: pd.DataFrame, lookahead=1, threshold=5.125): #5.125 pip works
    # Threshold methodology agnostic on pairs but actual value must be tuned.
    #Lookahead = 1 on static "just click" but can change if Arthur waits a time to wait longer to sell
    # etc
    """
    Create a 'TrueAction' column based on future returns.
    - BUY: price goes up > threshold
    - SELL: price goes down > threshold
    - HOLD: otherwise
    """

    
    cleaned_df["future_close"] = cleaned_df["close"].shift(-lookahead)
    cleaned_df["pip"] = (cleaned_df["future_close"] - cleaned_df["close"]) / pip_size(INSTRUMENT)

    def label_action(p):
        if p > threshold:
            return 0  # BUY
        elif p < -threshold:
            return 1  # SELL
        else:
            return 2  # HOLD
        


    cleaned_df["TrueAction"] = cleaned_df["pip"].apply(label_action)
    return cleaned_df.dropna(subset=["TrueAction"])

def verify_model_accuracy(df):
    """
    Print accuracy stats based on ActionHat vs TrueAction.
    """
    correct = (df["ActionHat"] == df["TrueAction"]).sum()
    total = len(df)
    accuracy = round(correct / total * 100, 2)
    print(f"✅ Model Accuracy: {accuracy}% ({correct}/{total} matches)")
    
    # Count of each class in TrueAction
    class_counts = df["TrueAction"].value_counts(normalize=True).sort_index()
    class_percentages = (class_counts * 100).round(4)

    print("📊 TrueAction Class Distribution:")
    print(f"  BUY (0):  {class_percentages.get(0, 0.000)}%")
    print(f"  SELL (1): {class_percentages.get(1, 0.000)}%")
    print(f"  HOLD (2): {class_percentages.get(2, 0.000)}%")

    tbt_confusion_matrix = confusion_matrix(df["TrueAction"],df["ActionHat"])
    print(tbt_confusion_matrix)

    



def run_historical_verification():
    current_date = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_date = datetime.strptime(END_DATE, "%Y-%m-%d")

    full_results = []

    while current_date <= end_date:
        print(f"Processing {current_date.date()}...")

        # Step 1: Extract
        candles , instrument_holder , granularity_holder = fetch_historical_candles(
            instrument=INSTRUMENT,
            date=current_date.strftime("%Y-%m-%d"),
            granularity=GRANULARITY
        )

        if not candles:
            print(f"⚠️ No candles returned on {current_date.date()}. Skipping.")
            current_date += timedelta(days=1)
            continue

        # Step 2-3: Clean
        tabular = convert_to_tabular(candles)
        cleaned, safe_timestamp = clean_tabular(tabular)

        # Step 4a: Label true outcomes
        labeled_df = generate_true_actions(cleaned, lookahead=TRUTH_LOOKAHEAD)
        # labeled_df has columns OHLCV , timestamp , complete and future close , return , TrueAction
        # a signal matrix has the columns "t_sec", "B", "S", "theta", "ActionHat"


        # Step 4b: Generate Signal Matrix and Classification
        signal_matrix = transform_to_signal(cleaned)


        # Step 5: Merge signal and truth on index (assuming aligned)
        verification_df = pd.DataFrame({
            "t_sec": signal_matrix["t_sec"],
            "value" : signal_matrix["value"],
            "volume": signal_matrix["volume"],
            "Buy" : signal_matrix["B"],
            "Sell": signal_matrix["S"],
            "margin": signal_matrix["margin"],
            "theta": signal_matrix["theta"],
            "mu": signal_matrix["mu"],
            "ActionHat": signal_matrix["ActionHat"],
            "TrueAction": labeled_df["TrueAction"].values  # use `.values` to avoid index mismatch
        })

        full_results.append(verification_df)
        current_date += timedelta(days=1)
        time.sleep(1)  # avoid throttling

    # Combine all days
    full_df = pd.concat(full_results, ignore_index=True)
    full_df.to_csv(f"verificationhistoricaldata/full_labeled_{INSTRUMENT}_{START_DATE}_to_{END_DATE}.csv", index=False)
    verify_model_accuracy(full_df)

    print(full_df["TrueAction"].value_counts())
    print(full_df["ActionHat"].value_counts())

if __name__ == "__main__":
    run_historical_verification()
