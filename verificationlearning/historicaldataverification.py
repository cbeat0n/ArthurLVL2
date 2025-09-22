### historicaldataverification.py is a small framework that allows for the DDT model to be
### tested in a deprecatable manner separate from Arthur's framework in order to
### allow for demo and then live testing without wasting resources. This is akin to 
### finding the y for a ML framework and looking for training accuracy, before demo test
### and live test accuracy. Last but not least something similar to this will be integrated 
### into load_signal or similar so that the predictive accuracy of Arthur can be upgraded
### in real time.

# External Dependencies
import pandas as pd
import time
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta
import sys
import os
import numpy as np
from sklearn.metrics import confusion_matrix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Internal Dependencies
from utils.transform_cleaned import convert_to_tabular, clean_tabular
from utils.transform_signal import transform_to_signal, decisionrule
from utils.load_signal import load_signal_matrix

# 1. Historical Date Range
START_DATE = "2025-09-01"
END_DATE = "2025-09-05"
INSTRUMENT = "USD_JPY"
INTAKE = "S30"
HORIZON = "M1"
ZONE = 8 # Given as integer for "MX".

#Fake Budget System - Capital , Buy size notional , incremented long units.

CAPITAL = 2500000
BLOTSIZE = 5000
LONGUNITS = 0


# keep this here because you start with demo testing before you ever go live. That's why
# pip size is in historicaldataverification.
def pip_size(symbol = "USD_JPY"):
    """
    Defines the size of a pip for an FX pair.
    """

    if symbol == "USD_JPY":
        return 0.01
    else:
        return 0.0001 # Generic pip and can be made for my pair options.

def fetch_historical_candles(instrument="USD_JPY", intgranularity="S5", price="M", date=None, timezone="UTC"):
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
            "granularity": intgranularity,
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

    return all_candles, instrument, intgranularity



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


def simulate_trades_no_cost(mySignal: pd.DataFrame) -> pd.DataFrame:
    """
    Causal, no-cost simulator:
    - BUY adds units (monopoly budget).
    - SELL dumps 0.95 of units; if 0 units, no-op.
    - HOLD does nothing.
    - Fills at *current* bar's price for MVP (or use t+1 if you prefer strict causality).
    """

    state_units = LONGUNITS
    entry_px = None
    side = None
    trades = []

    for _, r in mySignal.iterrows():
        px = float(r["value"])  # mid or close at horizon
        hat = r["ActionHat"]  # 0=BUY,1=SELL,2=HOLD

        # open/add long on BUY
        if hat == 0:
            # add units at px (book as new leg)
            units = BLOTSIZE
            state_units += units
            trades.append({"ts" : r["timestamp"],"side":"BUY","units" : units , "entry":px,"exit":None,"r":None})

        # dump on SELL
        elif hat == 1:
            if state_units > 0:
                to_close = min(0.90 * state_units , state_units)
                state_units -= to_close

                # go through BUY rows (FIFO) and decrement their units
                remaining = to_close
                for t in trades:
                    if t["side"] == "BUY" and t["units"] > 0:
                        take = min(t["units"], remaining)
                        t["units"] -= take
                        remaining -= take
                        if remaining <= 0:
                            break

                # recalc avg_entry from what's still open
                open_legs = [t for t in trades if t["side"]=="BUY" and t["units"] > 0]
                if open_legs:
                    avg_entry = sum(t["entry"] * t["units"] for t in open_legs) / sum(t["units"] for t in open_legs)
                    r_pnl = ((px - avg_entry) * to_close) / px
                    trades.append({"ts": r["timestamp"], "side":"CLOSE",
                                "units": to_close, "entry": avg_entry, "exit": px, "r": r_pnl})

        # HOLD => do nothing
        else:
            trades.append({"ts": r["timestamp"],"side": "HOLD","units": None, "entry": None , "exit": None , "r": None})

    return pd.DataFrame(trades)


def trade_vs_notrade_from_3x3(cm: np.ndarray) -> pd.DataFrame:
    # indices
    T = [0, 1]  # trade classes: BUY, SELL
    N = [2]     # no-trade class: HOLD

    TP = cm[np.ix_(T, T)].sum()
    FN = cm[np.ix_(T, N)].sum()
    FP = cm[np.ix_(N, T)].sum()
    TN = cm[np.ix_(N, N)].sum()

    df = pd.DataFrame(
        [[TP, FN],
         [FP, TN]],
        index=["TrueTrade", "TrueNoTrade"],
        columns=["PredTrade", "PredNoTrade"]
    )
    return df

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

    threedim_confusion_matrix = confusion_matrix(df["TrueAction"],df["ActionHat"] , labels = [0,1,2])
    twodim_confusion_matrix = trade_vs_notrade_from_3x3(threedim_confusion_matrix)
    busdays = np.busday_count(np.datetime64(START_DATE),np.datetime64(END_DATE)+np.timedelta64(1,"D"))
    n_trades = len(df[df["side"] == "CLOSE"])
    summary = {"p": np.median(df["p"]),
               "G": np.median(df["G"]),
               "L": np.median(df["L"]),
               "Navg": (n_trades)/(2*busdays),
               "D": busdays,
               "pl": sum(df["pl"])
               }

    mean = (summary["p"]*summary["G"]-(1-summary["p"])*summary["L"])*summary["Navg"]*summary["D"] # One instance for now
    sigma = ((summary["p"]*(1-summary["p"])*((summary["G"]+summary["L"])**2))*summary["Navg"]*summary["D"])**0.5

    sharpe = mean/sigma if sigma else float('inf') if mean > 0 else 0.0
    summary["sharpe"] = sharpe

    confidence_reg = 3.291 #99.9% CI 2S
    J_minus = mean - confidence_reg*sigma

    # Print out statement based on starting and ending capital, ending PnL, and J
    print("\n=== Decision 3x3 (B/S/H) ===\n", threedim_confusion_matrix)
    print("\n=== Trade vs No-Trade 2x2 ===\n", twodim_confusion_matrix)
    print("\n=== Weekly Stats ===\n", summary)
    # Start Capital, End Capital, PnL , J Statement, CI , Sharpe
    print(f"\n You started with {CAPITAL} and ended with {CAPITAL + summary['pl']} USD.")
    print(f"\n You produced {summary["pl"]} compared to minimum Bound on 99.9% CI 2S {J_minus}.")

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
            intgranularity=INTAKE
        )

        if not candles:
            print(f"⚠️ No candles returned on {current_date.date()}. Skipping.")
            current_date += timedelta(days=1)
            continue

        # Step 2-3: Clean
        tabular = convert_to_tabular(candles)
        cleaned, safe_timestamp = clean_tabular(tabular , horizon = ...)

        # Step 4a: Label true outcomes
        labeled_df = generate_true_actions(cleaned, lookahead = ... ) # get rid of lookahead
        # labeled_df has columns OHLCV , timestamp , complete and future close , return , TrueAction
        # a signal matrix has the columns "t_sec", "B", "S", "theta", "ActionHat"


        # Step 4b: Generate Signal Matrix and Classification
        signal_matrix = transform_to_signal(cleaned)


        # Step 4c: Make latency and cost free approximate trades using fake money.
        trades = simulate_trades_no_cost(signal_matrix)

        mask_close = trades["side"] == "CLOSE"
        rollwin = 50 # depends on horizon

        # Step 5: Merge signal and truth on index (assuming aligned)
        verification_df = pd.DataFrame({
            "ts": trades["ts"],
            "t_sec": signal_matrix["t_sec"],
            "value" : signal_matrix["value"],
            "volume": signal_matrix["volume"],
            "Buy" : signal_matrix["B"],
            "Sell": signal_matrix["S"],
            "margin": signal_matrix["margin"],
            "theta": signal_matrix["theta"],
            "mu": signal_matrix["mu"],
            "ActionHat": signal_matrix["ActionHat"],
            "TrueAction": labeled_df["TrueAction"].values,  # use `.values` to avoid index mismatch
            "side": trades["side"],
            "units": trades["units"],
            "entry": trades["entry"],
            "exit" : trades["exit"],
            "rawpnl": trades["r"],
            "p": (trades["r"] > 0).astype(float).where(mask_close).rolling(rollwin).mean(),
            "G": trades["r"].where((trades["r"] > 0) & mask_close).rolling(rollwin).median(),
            "L": (-trades["r"]).where((trades["r"] <= 0) & mask_close).rolling(rollwin).median(),
            "pl": trades["r"].where(mask_close).rolling(rollwin).sum(),
        })

        full_results.append(verification_df)
        current_date += timedelta(days=1)
        time.sleep(1)  # avoid throttling

    # Combine all days
    full_df = pd.concat(full_results, ignore_index=True)
    full_df.to_csv(f"verificationhistoricaldata/full_labeled_{INSTRUMENT}_{START_DATE}_to_{END_DATE}.csv", index=False)
    verify_model_accuracy(full_df)


    # print these to see actual values but less useful
    #print(full_df["TrueAction"].value_counts())
    #print(full_df["ActionHat"].value_counts())

if __name__ == "__main__":
    run_historical_verification()
