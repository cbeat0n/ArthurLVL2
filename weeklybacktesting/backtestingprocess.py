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
from utils.transform_signal import transform_to_signal, causal_signal
from utils.granularity import horizon_seconds

## Backtesting Constraints
START_DATE = "2025-09-01" # Start Date
END_DATE = "2025-09-05" # End Date
INSTRUMENT = "USD_JPY" # Instrument
INTAKE = "S30" # Intake
HORIZON = "M1Q0" # Horizon of form MXQN, e.g. M5Q2 for M5.5
ZONE = 10 # # Zone
CAUSAL = True # True: replay the live window per bar (no future data). False: one full-day signal (old, uses future data)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verificationhistoricaldata")

## Fake Budget System - Capital , Buy size notional , incremented long units.
## Data needs to be frozen to completely finalize this. This requires a data lake
## of cleaned data. Cleaned data at INTAKE X HORIZON for INSTRUMENT will be 
## a functionally infinite waterbed, but this needs to be taken from somewhere.
## This is not a MVP puzzle to solve. Very easy, takes time.
## It can also be taken from historical from data vendors, this also takes time.
## Not a MVP puzzle.

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

def fetch_historical_candles(instrument="USD_JPY", intake="S5", price="MBA", date=None, timezone="UTC"):
    """
    fetch_historical_candles fetches historical candles from OANDA for the day. 
    It returns a tuple.

    Inputs:
    instrument : str - instrument being traded
    intake : str - intake granularity
    price: str - any of M (mid), B (bid), A (ask), e.g. MBA for all three
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
            candles = [c for c in response.json().get("candles", []) if c.get("complete", True)]
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
    It returns a pandas DataFrame. The last row has no future close, so its TrueAction is NaN.

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
    
    cleaned_df["TrueAction"] = cleaned_df["pip"].apply(label_action).where(cleaned_df["pip"].notna())
    
    return cleaned_df


def quote_bars(candles: list, horizon: str) -> pd.DataFrame:
    """
    quote_bars builds bid and ask open and close prices at horizon, resampled exactly like
    clean_tabular so the rows line up with the signal bars.
    It returns a pandas DataFrame.

    Inputs:
    candles : list - OANDA candles fetched with bid and ask (price containing B and A)
    horizon : str - horizon of form MXQN
    """

    quotes = pd.DataFrame({
        "timestamp": pd.to_datetime([c["time"] for c in candles], utc=True),
        "bid_o": [float(c["bid"]["o"]) for c in candles], "bid_c": [float(c["bid"]["c"]) for c in candles],
        "ask_o": [float(c["ask"]["o"]) for c in candles], "ask_c": [float(c["ask"]["c"]) for c in candles],
    }).set_index("timestamp").sort_index()

    agg = {"bid_o": "first", "bid_c": "last", "ask_o": "first", "ask_c": "last"}
    return (quotes.resample(pd.Timedelta(horizon_seconds(horizon), "s"), label="right", closed="left")
                  .agg(agg).dropna().reset_index())


def simulate_trades(mySignal: pd.DataFrame) -> pd.DataFrame:
    """
    simulate_trades is the backtesting trade simulator, long only, pyramiding over a zone.
    BUY adds a BLOTSIZE lot while the long is under ZONE lots. The first SELL after a BUY dumps
    90% of the long FIFO (whole units), later SELLs in the same run do nothing until the next BUY.
    HOLD does nothing. A decision on bar t is filled at the open of bar t+1, buys at the ask and
    sells at the bid, so the spread is paid on every fill.
    It returns a pandas DataFrame with exactly one row per signal row.

    Inputs:
    mySignal : pd.DataFrame - the signal dataframe to trade on, with buy_px, sell_px and mark_px
    """

    ## One output row per signal row, so trades can be joined to the signal on index.
    cols = ["ts", "side", "units", "entry", "exit", "r", "open_units", "equity"]
    max_units = ZONE * BLOTSIZE # Pyramid cap
    lots = [] # Open long lots as [units, entry], FIFO
    armed = False # A BUY since the last dump, so the next SELL dumps
    realized = 0.0
    trades = []

    for _, r in mySignal.iterrows():
        hat = r["ActionHat"]  # 0=BUY,1=SELL,2=HOLD
        open_units = sum(u for u, _ in lots)
        row = {"ts": r["timestamp"], "side": "HOLD", "units": np.nan, "entry": np.nan, "exit": np.nan, "r": np.nan}

        ## No next bar to fill on
        if hat in (0, 1) and (pd.isna(r["buy_px"]) or pd.isna(r["sell_px"])):
            row["side"] = "NOFILL"

        ## Pyramid: add a lot at next bar's ask, up to ZONE lots
        elif hat == 0 and open_units + BLOTSIZE <= max_units:
            lots.append([BLOTSIZE, r["buy_px"]])
            armed = True
            row.update(side="BUY", units=BLOTSIZE, entry=r["buy_px"])

        elif hat == 0:
            armed = True
            row["side"] = "CAPPED"

        ## First SELL after buying: dump 90% at next bar's bid, whole units, FIFO
        elif hat == 1 and armed and open_units > 0:
            px = r["sell_px"]
            to_close = int(0.90 * open_units)

            ## Consume lots FIFO and accumulate the cost basis of the units actually closed
            remaining, closed_cost = to_close, 0.0
            while remaining > 0:
                take = min(lots[0][0], remaining)
                closed_cost += take * lots[0][1]
                lots[0][0] -= take
                remaining -= take
                if lots[0][0] <= 0:
                    lots.pop(0)

            avg_entry = closed_cost / to_close
            r_pnl = ((px - avg_entry) * to_close) / px
            realized += r_pnl
            armed = False
            row.update(side="CLOSE", units=to_close, entry=avg_entry, exit=px, r=r_pnl)

        ## Further SELLs in the same run, or SELL with nothing bought => stay put
        elif hat == 1:
            row["side"] = "NOOP"

        ## Mark the open long at this bar's closing bid, what closing it now would get
        open_units = sum(u for u, _ in lots)
        unrealized = sum(u * (r["mark_px"] - e) for u, e in lots) / r["mark_px"] if lots else 0.0
        row.update(open_units=open_units, equity=realized + unrealized)
        trades.append(row)

    return pd.DataFrame(trades, columns=cols, index=mySignal.index).astype(
        {"units": float, "entry": float, "exit": float, "r": float, "open_units": float, "equity": float})


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


def report(df : pd.DataFrame):
    """
    report prints trading performance first (net PnL after spread, expectancy, drawdown,
    per-day PnL), then the decision diagnostics. PnL is in account currency (USD).
    It returns a dict of the headline statistics.

    Inputs:
    df : pd.DataFrame - the verification dataframe from run_historical_verification
    """

    closes = df.loc[df["side"] == "CLOSE", "rawpnl"]
    equity = df["equity"]
    drawdown = (equity.cummax().clip(lower=0) - equity).max()
    ## Bars are labeled by their close, so the 23:59 bar is stamped 00:00. Group days by bar start.
    days = (df["ts"] - pd.Timedelta(horizon_seconds(HORIZON), "s")).dt.date
    spread_pips = ((df["ask_c"] - df["bid_c"]) / pip_size(INSTRUMENT)).median()

    stats = {"net_pnl": equity.iloc[-1], "realized": closes.sum(), "unrealized_end": equity.iloc[-1] - closes.sum(),
             "buys": int((df["side"] == "BUY").sum()), "closes": len(closes),
             "win_rate": (closes > 0).mean() if len(closes) else np.nan,
             "expectancy": closes.mean() if len(closes) else np.nan,
             "max_drawdown": drawdown, "median_spread_pips": spread_pips}

    ## Per-day PnL is the change in equity over the day, so open positions are counted
    daily = equity.groupby(days).last().diff().fillna(equity.groupby(days).last().iloc[0])
    per_day = pd.DataFrame({"pnl": daily, "closes": (df["side"] == "CLOSE").groupby(days).sum(),
                            "bars": df.groupby(days).size()})

    print("\n=== Trading Performance (USD, next-bar bid/ask fills) ===")
    for k, v in stats.items():
        print(f"  {k:20s} {v:,.4f}" if isinstance(v, float) else f"  {k:20s} {v}")
    print("\n=== Per Day ===\n", per_day.round(2))

    ## J statement from closed trades: p win rate, G median gain, L median loss
    if len(closes):
        D = days.nunique()
        p, G, L = (closes > 0).mean(), closes[closes > 0].median(), (-closes[closes <= 0]).median()
        G, L = (0.0 if pd.isna(G) else G), (0.0 if pd.isna(L) else L)
        Navg = len(closes) / (2 * D)
        mean = (p * G - (1 - p) * L) * Navg * D
        sigma = ((p * (1 - p) * ((G + L) ** 2)) * Navg * D) ** 0.5
        sharpe = mean / sigma if sigma else float('inf') if mean > 0 else 0.0
        J_minus = mean - 3.291 * sigma # 99.9% CI 2S
        print(f"\n p {p:.3f}  G {G:.2f}  L {L:.2f}  Navg {Navg:.2f}  D {D}  sharpe {sharpe:.3f}")
        print(f" You started with {CAPITAL} and ended with {CAPITAL + stats['net_pnl']:,.2f} USD.")
        print(f" You produced {stats['net_pnl']:,.2f} compared to minimum Bound on 99.9% CI 2S {J_minus:,.2f}.")
    else:
        print(f"\n No closed trades. You started with {CAPITAL} and ended with {CAPITAL + stats['net_pnl']:,.2f} USD.")

    ## Decision diagnostics, secondary. Last bar has no label.
    labeled = df.dropna(subset=["TrueAction"])
    cm = confusion_matrix(labeled["TrueAction"].astype(int), labeled["ActionHat"].astype(int), labels=[0, 1, 2])
    print("\n=== Decision 3x3 (B/S/H), diagnostic only ===\n", cm)
    print("\n=== Trade vs No-Trade 2x2 ===\n", trade_vs_notrade_from_3x3(cm))

    return stats


def run_historical_verification(start_date : str = START_DATE , end_date : str = END_DATE):
    """
    run_historical_verification runs a historical verification process, the proprietary backtesting
    process. Each day is fetched, cleaned and turned into a signal on its own, then the days are
    traded as one continuous account so positions carry overnight.
    It returns the verification DataFrame.

    Inputs:
    start_date : str - first day, YYYY-MM-DD
    end_date : str - last day, YYYY-MM-DD
    """
    
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    days = []

    while current_date <= end:
        print(f"Processing {current_date.date()}...")

        ## Mid for the signal, bid/ask for fills
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

        ## T2, causal keeps every bar; bars before the first full live window have no signal and HOLD
        if CAUSAL:
            causal = causal_signal(cleaned, ZONE)["signal"]
            signal = cleaned[["timestamp"]].merge(causal, on="timestamp", how="left")
            signal["ActionHat"] = signal["ActionHat"].fillna(2).astype(int)
        else:
            signal = transform_to_signal(io.StringIO(cleaned.to_csv(index = False)) , ZONE)["signal"]
            signal["timestamp"] = pd.to_datetime(signal["timestamp"], utc=True)
        signal = signal.merge(quote_bars(candles, HORIZON), on="timestamp", how="left")

        ## Label Outcomes, last bar of the day has no label
        days.append(generate_true_actions(signal))

        current_date += timedelta(days=1)
        time.sleep(0.225)  # Throttle cooler

    if not days:
        raise ValueError(f"No candles returned between {start_date} and {end_date}.")
    signal = pd.concat(days, ignore_index=True)

    ## Next-bar fills: decision on bar t fills at bar t+1's open, ask to buy, bid to sell.
    ## A decision on a day's last bar fills at the next trading day's first bar.
    signal["buy_px"] = signal["ask_o"].shift(-1)
    signal["sell_px"] = signal["bid_o"].shift(-1)
    signal["mark_px"] = signal["bid_c"]

    trades = simulate_trades(signal)

    verification_df = pd.DataFrame({
        "ts": signal["timestamp"],
        "t_sec": signal["T"],
        "value" : signal["value"],
        "volume": signal["volume"],
        "Buy" : signal["B"],
        "Sell": signal["S"],
        "margin": signal["Margin"],
        "theta": signal["Theta"],
        "mu": signal["Mu"],
        "ActionHat": signal["ActionHat"],
        "TrueAction": signal["TrueAction"],
        "bid_c": signal["bid_c"],
        "ask_c": signal["ask_c"],
        "side": trades["side"],
        "units": trades["units"],
        "entry": trades["entry"],
        "exit" : trades["exit"],
        "rawpnl": trades["r"],
        "open_units": trades["open_units"],
        "equity": trades["equity"],
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"full_labeled_{INSTRUMENT}_{HORIZON}_z{ZONE}_{start_date}_to_{end_date}.csv")
    verification_df.to_csv(out, index=False)
    print(f"Saved {len(verification_df)} bars to {out}")

    report(verification_df)
    return verification_df


## Adds modularity
if __name__ == "__main__":
    run_historical_verification()
