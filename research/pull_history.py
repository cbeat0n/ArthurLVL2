### pull_history.py pulls months of historical candles from OANDA's demo API for signal research.
### It is read only: it places no orders. One file per calendar month, so a stopped run picks up
### where it left off and a finished month is never fetched twice.

## External Dependencies
import os , sys , time # Libraries for operating system communication and time
from datetime import datetime , timedelta , timezone # Library for time objects
from dotenv import load_dotenv # Env hides the API key
import pandas as pd # Library for data manipulation

sys.path.insert(0 , os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # repo root, so utils imports work

## Internal Dependencies
from utils.http_helpers import OandaSession , json_or_raise
from utils.instruments import SUPPORTED

## What to pull. Whole minute horizons (M1Q0, M2Q0, M3Q0, M5Q0 ...) are all built from M1 candles.
INSTRUMENT = "USD_JPY"
GRANULARITY = "M1"
START_DATE = "2026-01-01" # first day, inclusive
END_DATE = "2026-10-31" # last day, inclusive
CHUNK_DAYS = 3 # M1 gives at most 4320 candles in 3 days, under OANDA's 5000 candle limit
THROTTLE_S = 0.225 # pause between requests, same as the backtest
BASE_URL = "https://api-fxpractice.oanda.com"
OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) , "data" , "history")

COLUMNS = ["timestamp" , "volume" ,
           "mid_o" , "mid_h" , "mid_l" , "mid_c" ,
           "bid_o" , "bid_h" , "bid_l" , "bid_c" ,
           "ask_o" , "ask_h" , "ask_l" , "ask_c"]


def candles_to_frame(candles : list):
    """
    candles_to_frame turns OANDA candles fetched with price MBA into one row per complete candle,
    sorted by time with duplicate timestamps dropped.
    It returns a pandas DataFrame with the columns in COLUMNS.

    Inputs:
    candles : list - OANDA candle dictionaries
    """

    rows = []
    for c in candles:
        if not c.get("complete" , True):
            continue
        row = {"timestamp" : pd.to_datetime(c["time"] , utc = True).strftime("%Y-%m-%d %H:%M:%S+00:00") ,
               "volume" : int(c["volume"])}
        for side , key in (("mid" , "mid") , ("bid" , "bid") , ("ask" , "ask")):
            for leg in ("o" , "h" , "l" , "c"):
                row[f"{side}_{leg}"] = float(c[key][leg])
        rows.append(row)

    df = pd.DataFrame(rows , columns = COLUMNS)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop = True)


def month_ranges(start_date : str , end_date : str):
    """
    month_ranges splits a date range into calendar months, each clipped to the range.
    It returns a list of (label, first_day, day_after_last) with datetimes in UTC.

    Inputs:
    start_date : str - first day, YYYY-MM-DD
    end_date : str - last day, YYYY-MM-DD, inclusive
    """

    first = datetime.strptime(start_date , "%Y-%m-%d").replace(tzinfo = timezone.utc)
    last = datetime.strptime(end_date , "%Y-%m-%d").replace(tzinfo = timezone.utc) + timedelta(days = 1)
    if last <= first:
        raise ValueError(f"END_DATE {end_date} is before START_DATE {start_date}.")

    out = []
    cursor = first
    while cursor < last:
        nxt = (cursor.replace(day = 1) + timedelta(days = 32)).replace(day = 1)
        out.append((cursor.strftime("%Y-%m") , cursor , min(nxt , last)))
        cursor = nxt
    return out


def fetch_range(http , instrument : str , granularity : str , begin : datetime , end : datetime):
    """
    fetch_range fetches all complete candles in [begin, end) in chunks of CHUNK_DAYS days.
    It returns a list of OANDA candle dictionaries.

    Inputs:
    http : OandaSession - an authenticated session
    instrument : str - the instrument
    granularity : str - the candle size, e.g. M1
    begin : datetime - first moment, inclusive
    end : datetime - last moment, exclusive
    """

    candles = []
    cursor = begin
    while cursor < end:
        stop = min(cursor + timedelta(days = CHUNK_DAYS) , end)
        params = {"from" : cursor.strftime("%Y-%m-%dT%H:%M:%SZ") ,
                  "to" : stop.strftime("%Y-%m-%dT%H:%M:%SZ") ,
                  "granularity" : granularity , "price" : "MBA" , "alignmentTimezone" : "UTC"}
        body = json_or_raise(http.get(f"{BASE_URL}/v3/instruments/{instrument}/candles" , params = params) ,
                             f"history {instrument} {cursor:%Y-%m-%d}")
        candles.extend(body.get("candles" , []))
        cursor = stop
        time.sleep(THROTTLE_S)
    return candles


def write_atomic(df : pd.DataFrame , path : str):
    """
    write_atomic writes a gzipped CSV so the file at path is always whole: a temporary file is
    written first and then swapped in.
    It returns the path.

    Inputs:
    df : pd.DataFrame - the rows to write
    path : str - the finished filepath, ending .csv.gz
    """

    os.makedirs(os.path.dirname(path) , exist_ok = True)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        df.to_csv(tmp , index = False , compression = "gzip")
        os.replace(tmp , path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def pull_history(instrument : str = INSTRUMENT , granularity : str = GRANULARITY ,
                 start_date : str = START_DATE , end_date : str = END_DATE , http = None):
    """
    pull_history pulls the range month by month into data/history/<instrument>_<granularity>/. A month
    already on disk is skipped, unless it is the current month, which is always fetched again.
    It returns a list of the monthly file paths.

    Inputs:
    instrument : str - the instrument, must be in SUPPORTED
    granularity : str - the candle size
    start_date : str - first day, YYYY-MM-DD
    end_date : str - last day, YYYY-MM-DD, inclusive
    http : OandaSession - for tests. By default a session using OANDADEMOAPIKEY from .env
    """

    if instrument not in SUPPORTED:
        raise ValueError(f"Unsupported instrument {instrument!r}. Supported: {', '.join(SUPPORTED)}")

    if http is None:
        load_dotenv()
        key = os.getenv("OANDADEMOAPIKEY")
        if not key:
            raise RuntimeError("OANDADEMOAPIKEY is not set. Put it in .env.")
        http = OandaSession()
        http.headers.update({"Authorization" : f"Bearer {key}"})

    this_month = datetime.now(timezone.utc).strftime("%Y-%m")
    folder = os.path.join(OUT_ROOT , f"{instrument}_{granularity}")
    paths = []

    for label , begin , end in month_ranges(start_date , end_date):
        path = os.path.join(folder , f"{instrument}_{granularity}_{label}.csv.gz")
        paths.append(path)
        if os.path.exists(path) and label != this_month:
            print(f"{label}: already on disk, skipped" , flush = True)
            continue
        df = candles_to_frame(fetch_range(http , instrument , granularity , begin , end))
        write_atomic(df , path)
        print(f"{label}: {len(df)} candles saved" , flush = True)

    return paths


def check_history(paths : list):
    """
    check_history prints a short integrity report over the monthly files: rows, first and last
    candle, the largest gap inside each month, and any rows where the bid is above the ask.
    It returns nothing.

    Inputs:
    paths : list - the monthly file paths
    """

    for p in paths:
        df = pd.read_csv(p)
        if df.empty:
            print(f"{os.path.basename(p)}: EMPTY")
            continue
        ts = pd.to_datetime(df["timestamp"] , utc = True)
        gaps = ts.diff().dropna()
        crossed = int((df["bid_c"] > df["ask_c"]).sum())
        print(f"{os.path.basename(p)}: {len(df)} rows, {ts.iloc[0]:%Y-%m-%d %H:%M} to {ts.iloc[-1]:%Y-%m-%d %H:%M}, "
              f"largest gap {gaps.max() if len(gaps) else 'n/a'}, crossed quotes {crossed}")


## Adds modularity
if __name__ == "__main__":
    check_history(pull_history())
