### backtestingprocess.py is a modular and agnostic framework that automates
### backtesting and sends data to be visualized so parameters and strategies
### can be tested. It is currently semisupervised, where TrueAction is defined
### with lookahead values, and other parameters are unsupervised, like B,S params
### and U,V, etc.

## External Dependencies
import pandas as pd # Library for data manipulation
import io # Library for string parsing
import time # Library for time
from dotenv import load_dotenv # Library for .env communication
import requests # Library for HTTP Requests
from datetime import datetime, timedelta # Library for time objects
import sys # Library for system communications
import os # Library for operating system communications
import numpy as np # Library for numerical caluclations
from sklearn.metrics import confusion_matrix # Library for data manipulation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


## Internal Dependencies
from utils.transform_cleaned import convert_to_tabular, clean_tabular
from utils.transform_signal import transform_to_signal, decisionrule
from utils.load_signal import load_signal_matrix

## Backtesting Constraints
START_DATE = "2025-09-01" # Start Date
END_DATE = "2025-09-05" # End Date
INSTRUMENT = "USD_JPY" # Instrument
INTAKE = "S30" # Intake
HORIZON = "M1Q0" # Horizon of form MXQN, e.g. M5Q2 for M5.5
ZONE = 10 # # Zone

## Fake Budget System - Capital , Buy size notional , incremented long units.
## Data needs to be frozen to completely finalize this. This requires a data lake
## of cleaned data. Cleaned data at INTAKE X HORIZON for INSTRUMENT will be 
## a functionally infinite waterbed, but this needs to be taken from somewhere.
## This is not a MVP puzzle to solve. Very easy, takes time.
## It can also be taken from historical from data vendors, this also takes time.
## Not a MVP puzzle. Bryan input also necessary.

CAPITAL = 2500000 # Initial Capital
BLOTSIZE = 5000 # Buying Lotsize
LONGUNITS = 0 # Long Units Start


def pip_size(symbol = "USD_JPY"):
    """
    pip_size defines the size of a pip for an FX pair.
    Strategies start as backtested, then go to demo, then to live. That is why this function
    is here and not elsewhere.
    It returns a float of the percentage in points
    
    Inputs:
    symbol : str - instrument symbol
    """

    if symbol == "USD_JPY":
        return 0.01
    else:
        return 0.0001 # Generic pip and can be made for other options.

def fetch_historical_candles(instrument="USD_JPY", intake="S5", price="M", date=None, timezone="UTC"):
    """
    fetch_historical_candles fetches historical candles from OANDA for the day. 
    It returns a tuple.

    Inputs:
    instrument : str - instrument being traded
    intake : str - intake granularity
    price: str - midpoint, bid, or ask prices
    date : str - date given to parse HTTP
    timezone: str - timezone of information
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

    ## Convert date string to datetime object
    start_time = datetime.strptime(date, "%Y-%m-%d")
    end_time = start_time + timedelta(days=1)

    ## Define safe window chunk size based on granularity
    ## For S5, safe limit is 4 hours
    chunk_size = timedelta(hours=4)

    all_candles = []
    current_time = start_time

    while current_time < end_time:
        chunk_start = current_time
        chunk_end = min(current_time + chunk_size, end_time)

        params = {
            "from": chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "granularity": intake,
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

    return all_candles, instrument, intake



def generate_true_actions(cleaned_df: pd.DataFrame, threshold=5.125):
    """
    generate_true_actions allows for supervised learning by making a definition of
    'make a trade here' and this is crossreferenced with ActionHat. Definition will
    need to be changed for more dynamic trading.
    It returns a pandas DataFrame.

    Inputs:
    cleaned_df : pd.DataFrame - the dataframe to define the target column on
    threshold : float - the number of pips to define as minimum threshold
    """

    
    cleaned_df["future_close"] = cleaned_df["close"].shift(-1) # Looks ahead 1 row for truth,
    # and needs to be made dynamic for a supervised learning methodology for
    # dynamic wait times and fill orders, NOT MVP.

    cleaned_df["pip"] = (cleaned_df["future_close"] - cleaned_df["close"]) / pip_size(INSTRUMENT)

    ## Helper function
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
    simulate_trades_no_cost is the MVP backtesting process. It supposes no latency, no costs
    on trading. This is mainly because OANDA keeps financial information hidden and baked in,
    such as costs. Slippage and otherwise is tracked in finalized records. For a more robust
    and later version, cost and slippage can be generated mathematically and data can be
    taken from frozen sets.
    It returns a pandas DataFrame.

    Inputs:
    mySignal : pd.DataFrame - the signal dataframe to trade on
    """

    ## Exactly one output row per signal row, so trades can be joined to the signal on index.
    cols = ["ts", "side", "units", "entry", "exit", "r"]
    state_units = LONGUNITS
    trades = []

    for _, r in mySignal.iterrows():
        px = float(r["value"])  # mid or close at horizon
        hat = r["ActionHat"]  # 0=BUY,1=SELL,2=HOLD

        ## Open/add long on BUY
        if hat == 0:

            ## Add units at px (book as new leg)
            units = BLOTSIZE
            state_units += units
            trades.append({"ts" : r["timestamp"],"side":"BUY","units" : units , "entry":px,"exit":np.nan,"r":np.nan})

        ## Dump on SELL
        elif hat == 1 and state_units > 0:
            to_close = min(0.90 * state_units , state_units)
            state_units -= to_close

            ## Go through BUY rows (FIFO), decrement their units and accumulate
            ## the cost basis of the units actually being closed.
            remaining = to_close
            closed_cost = 0.0
            for t in trades:
                if t["side"] == "BUY" and t["units"] > 0:
                    take = min(t["units"], remaining)
                    t["units"] -= take
                    closed_cost += take * t["entry"]
                    remaining -= take
                    if remaining <= 0:
                        break

            avg_entry = closed_cost / to_close
            r_pnl = ((px - avg_entry) * to_close) / px
            trades.append({"ts": r["timestamp"], "side":"CLOSE",
                        "units": to_close, "entry": avg_entry, "exit": px, "r": r_pnl})

        ## SELL while flat => nothing to close, still book a row to keep alignment
        elif hat == 1:
            trades.append({"ts": r["timestamp"],"side": "NOOP","units": np.nan, "entry": np.nan , "exit": np.nan , "r": np.nan})

        ## HOLD => do nothing
        else:
            trades.append({"ts": r["timestamp"],"side": "HOLD","units": np.nan, "entry": np.nan , "exit": np.nan , "r": np.nan})

    ## Numeric columns stay float even on days with no closes (all NaN).
    return pd.DataFrame(trades, columns=cols, index=mySignal.index).astype(
        {"units": float, "entry": float, "exit": float, "r": float})


def trade_vs_notrade_from_3x3(cm: np.ndarray) -> pd.DataFrame:
    """
    trade_vs_notrade_from_3x3 makes a 2x2 confusion matrix on whether or not there are trades
    made. This is not to be confused for the actualized G vs L values and PnL calculations.
    It returns a pandas DataFrame.

    Inputs:
    cm : np.ndarray - a confusion matrix in 3x3 dimensions
    """
    
    ## Indices
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

def verify_model_accuracy(df : pd.DataFrame , pl : float):
    """
    verify_model_accuracy prints accuracy stats.
    It returns nothing.

    Inputs:
    df : pd.DataFrame - the dataframe to verify accuracies on
    pl : float - the pl of the timeframe
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
               "pl": pl
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
    print(f"\n You produced {summary['pl']} compared to minimum Bound on 99.9% CI 2S {J_minus}.")

def run_historical_verification():
    """
    run_historical_verification runs a historical verification process, the proprietary backtesting
    process.
    It returns nothing.
    """
    
    current_date = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_date = datetime.strptime(END_DATE, "%Y-%m-%d")

    full_results = []

    while current_date <= end_date:
        print(f"Processing {current_date.date()}...")

        candles , instrument_holder , granularity_holder = fetch_historical_candles(
            instrument=INSTRUMENT,
            date=current_date.strftime("%Y-%m-%d"),
            intake=INTAKE
        )

        if not candles:
            print(f"⚠️ No candles returned on {current_date.date()}. Skipping.")
            current_date += timedelta(days=1)
            continue

        ## T1
        tabular = convert_to_tabular({"candles": candles})
        cleaned = clean_tabular(tabular , horizon = HORIZON , intake = INTAKE)["cleaned"]

        ## Label Outcomes
        labeled_df = generate_true_actions(cleaned) 
        
        ## T2
        signal_matrix = transform_to_signal(io.StringIO(cleaned.to_csv(index = False)) , ZONE)["signal"]

        ## Make trades, no cost no latency. With Frozen data, add costs and slippage.
        trades = simulate_trades_no_cost(signal_matrix)
        assert len(trades) == len(signal_matrix) == len(labeled_df), "trades/signal/labels misaligned"

        mask_close = trades["side"] == "CLOSE"
        rollwin = 5 # Rolling window of horizon

        dailypnl = []

        ## Merge on index over timestamp.
        verification_df = pd.DataFrame({
            "ts": trades["ts"],
            "t_sec": signal_matrix["T"],
            "value" : signal_matrix["value"],
            "volume": signal_matrix["volume"],
            "Buy" : signal_matrix["B"],
            "Sell": signal_matrix["S"],
            "margin": signal_matrix["Margin"],
            "theta": signal_matrix["Theta"],
            "mu": signal_matrix["Mu"],
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
            "pl": trades["r"].where(mask_close).fillna(0.0).cumsum(),
        })

        ## Last bar has no future_close, so its TrueAction is not a real label.
        verification_df = verification_df.iloc[:-1]

        full_results.append(verification_df)
        dailypnl.append(verification_df["pl"].iloc[-1])
        current_date += timedelta(days=1)
        time.sleep(0.225)  # Throttle cooler

    ## Make full dataframe
    full_df = pd.concat(full_results, ignore_index=True)
    full_df.to_csv(f"verificationhistoricaldata/full_labeled_{INSTRUMENT}_{START_DATE}_to_{END_DATE}.csv", index=False)
    verify_model_accuracy(full_df , sum(dailypnl))


## Adds modularity
if __name__ == "__main__":
    run_historical_verification()
