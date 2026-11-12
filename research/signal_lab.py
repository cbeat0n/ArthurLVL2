### signal_lab.py is the research harness for Arthur's signal. It builds horizon bars from the pulled
### history, replays the signal causally, labels what happened next, learns Theta x Mu regions on past
### months, scores them on a later month, and compares them with simple baselines after the spread.
### Every run is appended to research/trials.csv so the number of tries behind a result is on record.
### Research only: it places no orders. Months on or after HOLDOUT_START are refused unless asked for.

## External Dependencies
import argparse , glob , math , os , sys , time # Libraries for files, arguments and timing
from datetime import datetime , timezone # Library for time objects
from multiprocessing import Pool # Library for using every core
import numpy as np # Library for numerical work
import pandas as pd # Library for data manipulation

sys.path.insert(0 , os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # repo root, so utils imports work

## Internal Dependencies
from research.regions import learn_regions , apply_regions , usable , SPREAD_PIPS
from utils.instruments import pip_size
from utils.transform_signal import transform_to_signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_ROOT = os.path.join(ROOT , "data" , "history")
TRIALS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)) , "trials.csv")
HOLDOUT_START = pd.Timestamp("2026-08-01" , tz = "UTC") # Aug-Oct 2026 is sealed for one final look
WINDOW_PER_ZONE = 8 # live computes the signal on its last 8 * zone horizon bars
MAX_SIGNALS = 10000 # cap on signals per horizon, bars are skipped evenly to stay under it
AGG = {"open" : "first" , "high" : "max" , "low" : "min" , "close" : "last" , "volume" : "sum"}


def load_history(instrument : str = "USD_JPY" , granularity : str = "S5" , root : str = None ,
                 allow_holdout : bool = False):
    """
    load_history reads the monthly candle files written by pull_history.py and returns mid prices.
    Candles on or after HOLDOUT_START are dropped unless allow_holdout is True.
    It returns a DataFrame indexed by UTC timestamp with open, high, low, close, volume.

    Inputs:
    instrument : str - the instrument
    granularity : str - candle size of the files, e.g. S5 or M1
    root : str - folder holding <instrument>_<granularity>/, defaults to data/history
    allow_holdout : bool - True lets sealed months through, for the one final look only
    """

    folder = os.path.join(root or HISTORY_ROOT , f"{instrument}_{granularity}")
    files = sorted(glob.glob(os.path.join(folder , "*.csv.gz")))
    if not files:
        raise FileNotFoundError(f"No history files in {folder}. Run research/pull_history.py first.")

    cols = ["timestamp" , "volume" , "mid_o" , "mid_h" , "mid_l" , "mid_c"]
    df = pd.concat([pd.read_csv(f , usecols = cols) for f in files] , ignore_index = True)
    df["timestamp"] = pd.to_datetime(df["timestamp"] , utc = True)
    df = df.sort_values("timestamp").set_index("timestamp")
    if not allow_holdout:
        df = df[df.index < HOLDOUT_START]

    return df.rename(columns = {"mid_o" : "open" , "mid_h" : "high" , "mid_l" : "low" , "mid_c" : "close"})


def horizon_bars(candles : pd.DataFrame , seconds : int):
    """
    horizon_bars builds horizon bars the way clean_tabular does: label on the right edge, each bar
    closed on the left, so no bar contains a candle from its own future.
    It returns a DataFrame with timestamp, open, high, low, close, volume.

    Inputs:
    candles : pd.DataFrame - output of load_history
    seconds : int - horizon length in seconds, e.g. 1800 for M30Q0
    """

    return (candles.resample(pd.Timedelta(seconds = seconds) , label = "right" , closed = "left")
            .agg(AGG).dropna(how = "any").reset_index())


_BARS , _ZONE = None , None


def _init_worker(bars : pd.DataFrame , zone : int):
    ## Windows starts workers fresh instead of copying this process, so they are handed the bars here.
    global _BARS , _ZONE
    _BARS , _ZONE = bars , zone


def _signal_at(i : int):
    window = _BARS.iloc[i - WINDOW_PER_ZONE * _ZONE + 1 : i + 1].reset_index(drop = True)
    row = transform_to_signal(window , _ZONE)["signal"].iloc[-1]
    return float(row["Theta"]) , float(row["Mu"]) , float(row["Margin"])


def causal_signals(bars : pd.DataFrame , zone : int , max_signals : int = MAX_SIGNALS , workers : int = 2):
    """
    causal_signals replays the live engine at every k-th bar: the signal is computed on the trailing
    8 * zone bars and only the last row is kept, so nothing at bar t depends on a later bar.
    It returns a DataFrame with bar index i, timestamp, Theta, Mu and Margin. The bars it skips are
    spread evenly so that no more than max_signals are computed.

    Inputs:
    bars : pd.DataFrame - horizon bars
    zone : int - the zone
    max_signals : int - cap on signals computed
    workers : int - processes to use
    """

    first = WINDOW_PER_ZONE * zone - 1
    last = len(bars) - 1
    if last < first:
        raise ValueError("Not enough bars for one window.")

    stride = max(1 , math.ceil((last - first + 1) / max_signals))
    idx = list(range(first , last + 1 , stride))

    if workers > 1:
        with Pool(workers , initializer = _init_worker , initargs = (bars , zone)) as pool:
            res = pool.map(_signal_at , idx , chunksize = 200)
    else:
        _init_worker(bars , zone)
        res = [_signal_at(i) for i in idx]

    out = pd.DataFrame(res , columns = ["Theta" , "Mu" , "Margin"])
    out.insert(0 , "i" , idx)
    out.insert(1 , "timestamp" , bars["timestamp"].iloc[idx].to_numpy())
    return out


def add_outcomes(sig : pd.DataFrame , bars : pd.DataFrame , instrument : str = "USD_JPY" ,
                 pairs : list = None , horizon_bars_n : int = 1 , touch_bars : int = 6):
    """
    add_outcomes labels every signal with what happened next, in pips, before the spread. Entry is the
    close of the signal bar. Columns added:
      ret0 - the signal bar's own move
      long_pips / short_pips - the next-bar result: the close horizon_bars_n bars later minus entry
      for each (target , stop) pair: long_T{t}_S{s} and short_T{t}_S{s} - the result of a trade that exits at
      the target or the stop, whichever the later bars' high / low reach first within touch_bars bars,
      or at the last bar's close if neither is reached. A bar that reaches both counts as the stop.
    It returns sig with those columns.

    Inputs:
    sig : pd.DataFrame - output of causal_signals
    bars : pd.DataFrame - the same horizon bars
    instrument : str - sets the pip size
    pairs : list - (target , stop) pairs in pips, e.g. [(10 , 5)]
    horizon_bars_n : int - bars ahead for the next-bar label
    touch_bars : int - most bars a target / stop trade is held
    """

    pip = pip_size(instrument)
    close , high , low = bars["close"].to_numpy() , bars["high"].to_numpy() , bars["low"].to_numpy()
    idx = sig["i"].to_numpy()
    out = sig.copy()

    out["ret0"] = (close[idx] - close[np.maximum(idx - 1 , 0)]) / pip
    ahead = idx + horizon_bars_n
    valid = ahead < len(close)
    nxt = np.full(len(idx) , np.nan)
    nxt[valid] = (close[ahead[valid]] - close[idx[valid]]) / pip
    out["long_pips"] , out["short_pips"] = nxt , -nxt

    for target , stop in (pairs or []):
        longs , shorts = np.full(len(idx) , np.nan) , np.full(len(idx) , np.nan)
        for n , i in enumerate(idx):
            end = min(i + touch_bars , len(close) - 1)
            if end <= i or i + touch_bars > len(close) - 1:
                continue # the trade would run past the data, so it has no label
            entry = close[i]
            longs[n] = _touch(entry , high[i + 1 : end + 1] , low[i + 1 : end + 1] , close[end] , target , stop , pip , +1)
            shorts[n] = _touch(entry , high[i + 1 : end + 1] , low[i + 1 : end + 1] , close[end] , target , stop , pip , -1)
        out[f"long_T{target}_S{stop}"] , out[f"short_T{target}_S{stop}"] = longs , shorts
    return out


def _touch(entry , highs , lows , last_close , target , stop , pip , side):
    for h , l in zip(highs , lows):
        up , dn = (h - entry) / pip , (entry - l) / pip # how far the bar went each way
        hit_target = (up if side > 0 else dn) >= target
        hit_stop = (dn if side > 0 else up) >= stop
        if hit_stop:
            return -float(stop) # both in one bar counts as the stop
        if hit_target:
            return float(target)
    return side * (last_close - entry) / pip


def score(df : pd.DataFrame , actions : np.ndarray , long_col : str = "long_pips" ,
          short_col : str = "short_pips" , spread : float = SPREAD_PIPS):
    """
    score is the result of a rule over a set of rows, after the spread paid on each trade.
    It returns a dictionary: trades, hit rate (share of trades with a positive gross result),
    mean gross pips, mean net pips and total net pips.

    Inputs:
    df : pd.DataFrame - rows with the outcome columns
    actions : np.ndarray - 0 BUY , 1 SELL , 2 HOLD per row
    long_col : str - outcome of a long trade
    short_col : str - outcome of a short trade
    spread : float - round-trip cost in pips
    """

    actions = np.asarray(actions)
    gross = np.where(actions == 0 , df[long_col].to_numpy() , np.where(actions == 1 , df[short_col].to_numpy() , np.nan))
    gross = gross[~np.isnan(gross)]
    if len(gross) == 0:
        return {"trades" : 0 , "hit" : np.nan , "gross" : np.nan , "net" : np.nan , "net_total" : 0.0}
    return {"trades" : int(len(gross)) , "hit" : float((gross > 0).mean()) , "gross" : float(gross.mean()) ,
            "net" : float(gross.mean() - spread) , "net_total" : float((gross - spread).sum())}


def baselines(df : pd.DataFrame , long_col : str = "long_pips" , short_col : str = "short_pips" ,
              spread : float = SPREAD_PIPS):
    """
    baselines scores the rules a region rule has to beat: always buy, always sell and fade the last bar.
    It returns a dictionary of name -> score.

    Inputs:
    df : pd.DataFrame - rows with the outcome columns and ret0
    long_col , short_col , spread : see score
    """

    n = len(df)
    fade = np.where(df["ret0"].to_numpy() > 0 , 1 , np.where(df["ret0"].to_numpy() < 0 , 0 , 2))
    return {"always_buy" : score(df , np.zeros(n , dtype = int) , long_col , short_col , spread) ,
            "always_sell" : score(df , np.ones(n , dtype = int) , long_col , short_col , spread) ,
            "fade_last_bar" : score(df , fade , long_col , short_col , spread)}


def month_index(ts : pd.Series):
    """
    month_index numbers calendar months as year * 12 + month so months can be compared and counted.
    It returns an integer Series.

    Inputs:
    ts : pd.Series - UTC timestamps
    """

    return ts.dt.year * 12 + ts.dt.month


def walk_forward(df : pd.DataFrame , shape : tuple = (2 , 3) , train_months : int = 7 , gap_months : int = 1 ,
                 min_train_months : int = 3 , long_col : str = "long_pips" , short_col : str = "short_pips" ,
                 spread : float = SPREAD_PIPS , **learn_kwargs):
    """
    walk_forward learns regions on past months and scores them on a later month, month after month. For a
    test month T the training months are T - gap_months - train_months to T - gap_months - 1, so no month
    from the last gap_months months before T is used in training. gap_months = 0 scores the blind month, the
    month right after the training window. gap_months = 1 (default) skips a month, which is how a model
    trained through month t - 2 does in month t, where it is deployed. A test month with fewer than
    min_train_months training months is skipped, which lets the window grow at the start of the data.
    It returns (per-month DataFrame, pooled results dictionary). The pooled results hold the region rule and
    each baseline over all test months.

    Inputs:
    df : pd.DataFrame - signals with outcomes and timestamp
    shape : tuple - (Theta bins , Mu bins)
    train_months : int - longest training window
    gap_months : int - months skipped between training and the test month
    min_train_months : int - fewest training months allowed
    long_col , short_col , spread : see score
    learn_kwargs : passed to learn_regions
    """

    df = usable(df).dropna(subset = [long_col , short_col]).reset_index(drop = True)
    months = month_index(df["timestamp"])
    rows , tests = [] , []

    for T in sorted(months.unique()):
        lo , hi = T - gap_months - train_months , T - gap_months - 1
        train = df[(months >= lo) & (months <= hi)]
        have = months[(months >= lo) & (months <= hi)].nunique()
        test = df[months == T]
        if have < min_train_months or len(test) == 0:
            continue

        fit = train[["Theta" , "Mu"]].assign(long_pips = train[long_col] , short_pips = train[short_col])
        model = learn_regions(fit , shape = shape , spread = spread , **learn_kwargs)
        actions = apply_regions(test , model)
        res = score(test , actions , long_col , short_col , spread)
        rows.append({"month" : f"{(T - 1) // 12}-{(T - 1) % 12 + 1:02d}" , "train_months" : int(have) ,
                     "n_test" : len(test) , "traded" : res["trades"] , "hit" : res["hit"] ,
                     "gross" : res["gross"] , "net" : res["net"]})
        tests.append((test , actions))

    if not tests:
        return pd.DataFrame(rows) , {}

    pooled_df = pd.concat([t for t , _ in tests] , ignore_index = True)
    pooled_actions = np.concatenate([a for _ , a in tests])
    pooled = {"regions" : score(pooled_df , pooled_actions , long_col , short_col , spread)}
    pooled.update(baselines(pooled_df , long_col , short_col , spread))
    return pd.DataFrame(rows) , pooled


def log_trial(settings : dict , results : dict , path : str = TRIALS_PATH):
    """
    log_trial appends one row to the trial log: when, the settings and the pooled results of the rules
    that were scored. The log is what tells us how many tries sit behind any number.
    It returns the path.

    Inputs:
    settings : dict - everything that was set for the run
    results : dict - name -> score dictionary
    path : str - the CSV to append to
    """

    row = {"when_utc" : datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}
    row.update({f"set_{k}" : v for k , v in settings.items()})
    for name , res in results.items():
        for k , v in res.items():
            row[f"{name}_{k}"] = round(v , 4) if isinstance(v , float) else v

    new = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        new = pd.concat([old , new] , ignore_index = True)
    tmp = f"{path}.{os.getpid()}.tmp"
    new.to_csv(tmp , index = False)
    os.replace(tmp , path)
    return path


def run(horizon_min : int = 30 , zone : int = 5 , shape : tuple = (2 , 3) , granularity : str = "S5" ,
        pairs : list = None , touch_bars : int = 6 , train_months : int = 7 , gap_months : int = 1 ,
        instrument : str = "USD_JPY" , root : str = None , workers : int = 2 , log : bool = True ,
        spread : float = SPREAD_PIPS):
    """
    run does the whole experiment for one horizon: bars, causal signals, outcomes, then a walk-forward for the
    next-bar label and for each target / stop pair, with baselines. Aug-Oct 2026 is never loaded.
    It returns a dictionary of name -> (per-month DataFrame, pooled results).

    Inputs:
    horizon_min : int - whole-minute horizon, M{n}Q0
    zone : int - the zone
    shape : tuple - (Theta bins , Mu bins)
    granularity : str - which pulled history to read
    pairs : list - (target , stop) pairs in pips
    touch_bars : int - most bars a target / stop trade is held
    train_months , gap_months : int - walk-forward schedule
    instrument : str - the instrument
    root : str - history folder, defaults to data/history
    workers : int - processes for the signal replay
    log : bool - append to trials.csv
    spread : float - round-trip cost in pips, 0 shows the result before costs
    """

    t0 = time.time()
    candles = load_history(instrument , granularity , root)
    bars = horizon_bars(candles , 60 * horizon_min)
    sig = causal_signals(bars , zone , workers = workers)
    sig = add_outcomes(sig , bars , instrument , pairs , touch_bars = touch_bars)

    out = {}
    jobs = [("next_bar" , "long_pips" , "short_pips")]
    jobs += [(f"T{t}_S{s}" , f"long_T{t}_S{s}" , f"short_T{t}_S{s}") for t , s in (pairs or [])]
    for name , lc , sc in jobs:
        monthly , pooled = walk_forward(sig , shape , train_months , gap_months , long_col = lc , short_col = sc , spread = spread)
        out[name] = (monthly , pooled)
        if log and pooled:
            log_trial({"horizon_min" : horizon_min , "zone" : zone , "shape" : f"{shape[0]}x{shape[1]}" ,
                       "label" : name , "granularity" : granularity , "train_months" : train_months ,
                       "gap_months" : gap_months , "touch_bars" : touch_bars , "spread" : spread , "signals" : len(sig) ,
                       "seconds" : round(time.time() - t0)} , pooled)
    return out


def retrain(horizon_min : int = 30 , zone : int = 5 , shape : tuple = (2 , 3) , granularity : str = "S5" ,
            label_pair : tuple = None , touch_bars : int = 6 , train_months : int = 7 ,
            min_blind_trades : int = 30 , out_path : str = None , instrument : str = "USD_JPY" ,
            root : str = None , workers : int = 2 , allow_holdout : bool = False):
    """
    retrain is the monthly command. At the start of month t the newest month in the data is t - 1. It fits regions on
    the train_months months before that (t - 8 to t - 2 by default), scores them on the newest
    month in the data (the blind month, never used in fitting), and saves the model to out_path only if that
    blind month traded at least min_blind_trades times with a positive net result after the spread. If not,
    the saved model is left alone. The saved file records the window it was trained on.
    It returns a dictionary: saved, blind month result, baselines.

    Inputs:
    horizon_min , zone , shape , granularity , touch_bars , instrument , root , workers : see run
    label_pair : tuple - (target , stop) in pips to train on, None trains on the next-bar result
    train_months : int - months of training, ending the month before the blind month
    min_blind_trades : int - fewest blind-month trades that count as evidence
    out_path : str - where to save, defaults to models/model_params.json
    allow_holdout : bool - True includes Aug-Oct 2026, for the one final look only
    """

    from research.regions import save_model
    from utils.regions import MODEL_PATH

    candles = load_history(instrument , granularity , root , allow_holdout)
    bars = horizon_bars(candles , 60 * horizon_min)
    pairs = [label_pair] if label_pair else None
    sig = add_outcomes(causal_signals(bars , zone , workers = workers) , bars , instrument , pairs , touch_bars = touch_bars)
    lc , sc = ("long_pips" , "short_pips") if not label_pair else (f"long_T{label_pair[0]}_S{label_pair[1]}" , f"short_T{label_pair[0]}_S{label_pair[1]}")

    df = usable(sig).dropna(subset = [lc , sc]).reset_index(drop = True)
    months = month_index(df["timestamp"])
    blind = int(months.max())
    lo , hi = blind - train_months , blind - 1
    train , test = df[(months >= lo) & (months <= hi)] , df[months == blind]
    fit = train[["Theta" , "Mu"]].assign(long_pips = train[lc] , short_pips = train[sc])
    model = learn_regions(fit , shape = shape)
    res = score(test , apply_regions(test , model) , lc , sc)

    ok = res["trades"] >= min_blind_trades and res["net"] > 0
    if ok:
        save_model(model , out_path or MODEL_PATH ,
                   {"instrument" : instrument , "horizon_min" : horizon_min , "zone" : zone , "granularity" : granularity ,
                    "label" : lc , "train_from_month" : int(lo) , "train_to_month" : int(hi) ,
                    "blind_month" : blind , "blind_result" : res})
    return {"saved" : ok , "blind" : res , "baselines" : baselines(test , lc , sc)}


def show(out : dict):
    """
    show prints the pooled results of run() as one table.
    It returns nothing.

    Inputs:
    out : dict - the output of run
    """

    for name , (monthly , pooled) in out.items():
        print(f"\n== {name} ==")
        if not pooled:
            print("no test months (not enough training months yet)")
            continue
        print(pd.DataFrame(pooled).T.round(3).to_string())
        print("per month:"); print(monthly.round(3).to_string(index = False))


## Adds modularity
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description = "Walk-forward test of Theta x Mu regions. Never loads Aug-Oct 2026.")
    ap.add_argument("--horizon-min" , type = int , default = 30)
    ap.add_argument("--zone" , type = int , default = 5)
    ap.add_argument("--shape" , default = "2,3" , help = "Theta bins,Mu bins")
    ap.add_argument("--granularity" , default = "S5")
    ap.add_argument("--pairs" , default = "" , help = "target/stop pairs in pips, e.g. 10/5,8/4")
    ap.add_argument("--touch-bars" , type = int , default = 6)
    ap.add_argument("--train-months" , type = int , default = 7)
    ap.add_argument("--gap-months" , type = int , default = 1)
    ap.add_argument("--spread" , type = float , default = SPREAD_PIPS , help = "round-trip cost in pips, 0 shows the result before costs")
    ap.add_argument("--root" , default = None)
    args = ap.parse_args()
    prs = [tuple(float(x) if "." in x else int(x) for x in p.split("/")) for p in args.pairs.split(",") if p]
    show(run(args.horizon_min , args.zone , tuple(int(x) for x in args.shape.split(",")) , args.granularity ,
             prs , args.touch_bars , args.train_months , args.gap_months , root = args.root , spread = args.spread))
