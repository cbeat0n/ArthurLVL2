### regions.py learns the Theta x Mu region map that utils/regions.py reads at run time.
### Each region gets BUY , SELL or HOLD from how the outcomes in that region turned out on
### the training months, after the spread. Thin regions are pulled toward zero so a lucky corner
### with a few dozen bars can't win. Research only: it places no orders.

## External Dependencies
import json , os , sys # Libraries for the model file and imports
import numpy as np # Library for numerical work
import pandas as pd # Library for data manipulation

sys.path.insert(0 , os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # repo root, so utils imports work

## Internal Dependencies
from utils.regions import BUY , SELL , HOLD , SENTINEL , region_id

SPREAD_PIPS = 1.6 # median USD_JPY spread measured Jan-Jul 2026, paid once per round trip
MIN_N = 150 # fewest training bars a region needs before it may trade
SHRINK_K = 100 # a region's mean outcome is scaled by n / (n + SHRINK_K)
MARGIN_PIPS = 0.0 # expected net pips a region must clear to trade


def usable(df : pd.DataFrame):
    """
    usable keeps the rows with a real Theta and Mu: not missing and not the sentinel.
    It returns a DataFrame.

    Inputs:
    df : pd.DataFrame - rows with Theta and Mu columns
    """

    ok = df["Theta"].notna() & df["Mu"].notna()
    ok &= (df["Theta"] - SENTINEL).abs() > 1e-12
    ok &= (df["Mu"] - SENTINEL).abs() > 1e-12
    return df[ok]


def quantile_cuts(values : pd.Series , bins : int):
    """
    quantile_cuts returns the cut points that split values into equal-count bins.
    It returns a sorted list of bins - 1 floats, with duplicates removed.

    Inputs:
    values : pd.Series - training values
    bins : int - how many bins
    """

    if bins <= 1:
        return []
    qs = np.linspace(0 , 1 , bins + 1)[1:-1]
    return sorted({float(c) for c in values.quantile(qs)})


def learn_regions(train : pd.DataFrame , shape : tuple = (2 , 3) , spread : float = SPREAD_PIPS ,
                  min_n : int = MIN_N , shrink_k : float = SHRINK_K , margin : float = MARGIN_PIPS):
    """
    learn_regions cuts the Theta x Mu plane into shape[0] x shape[1] equal-count regions and labels each
    one. For a region with n bars, long outcome mean L and short outcome mean S in pips, the shrunk
    means are L * n / (n + shrink_k) and S * n / (n + shrink_k). The region is BUY if its shrunk
    L - spread clears margin and beats SELL, SELL if its shrunk S - spread does, and HOLD otherwise
    or if it has fewer than min_n bars.
    It returns a model dictionary that utils/regions.py can load.

    Inputs:
    train : pd.DataFrame - training rows with Theta , Mu , long_pips and short_pips
    shape : tuple - (Theta bins , Mu bins), 4 to 8 regions is the plan
    spread : float - round-trip cost in pips
    min_n : int - fewest bars a region needs to trade
    shrink_k : float - shrinkage strength
    margin : float - expected net pips a region must clear
    """

    df = usable(train).dropna(subset = ["long_pips" , "short_pips"])
    if len(df) == 0:
        raise ValueError("No usable training rows.")

    theta_cuts = quantile_cuts(df["Theta"] , shape[0])
    mu_cuts = quantile_cuts(df["Mu"] , shape[1])
    stub = {"theta_cuts" : theta_cuts , "mu_cuts" : mu_cuts}
    ids = np.array([region_id(t , m , stub) for t , m in zip(df["Theta"] , df["Mu"])])

    n_regions = (len(theta_cuts) + 1) * (len(mu_cuts) + 1)
    actions , cells = [] , []
    for rid in range(n_regions):
        sel = ids == rid
        n = int(sel.sum())
        if n == 0:
            actions.append(HOLD)
            cells.append({"region" : rid , "n" : 0 , "action" : HOLD})
            continue

        shrink = n / (n + shrink_k)
        long_mean = float(df["long_pips"][sel].mean())
        short_mean = float(df["short_pips"][sel].mean())
        exp_buy = long_mean * shrink - spread
        exp_sell = short_mean * shrink - spread

        action = HOLD
        if n >= min_n and max(exp_buy , exp_sell) > margin:
            action = BUY if exp_buy >= exp_sell else SELL

        actions.append(action)
        cells.append({"region" : rid , "n" : n , "long_mean_pips" : round(long_mean , 4) ,
                      "short_mean_pips" : round(short_mean , 4) , "exp_buy_net" : round(exp_buy , 4) ,
                      "exp_sell_net" : round(exp_sell , 4) , "action" : action})

    return {"version" : 1 , "theta_cuts" : theta_cuts , "mu_cuts" : mu_cuts , "actions" : actions ,
            "cells" : cells , "settings" : {"shape" : list(shape) , "spread" : spread , "min_n" : min_n ,
                                           "shrink_k" : shrink_k , "margin" : margin} ,
            "trained_rows" : int(len(df))}


def apply_regions(df : pd.DataFrame , model : dict):
    """
    apply_regions gives the action for every row, using the same lookup as live.
    It returns an integer array of 0 BUY , 1 SELL , 2 HOLD. Rows without a usable Theta and Mu are HOLD.

    Inputs:
    df : pd.DataFrame - rows with Theta and Mu
    model : dict - a learned model
    """

    out = np.full(len(df) , HOLD , dtype = int)
    for i , (t , m) in enumerate(zip(df["Theta"].to_numpy() , df["Mu"].to_numpy())):
        rid = region_id(t , m , model)
        if rid is not None:
            out[i] = model["actions"][rid]
    return out


def save_model(model : dict , path : str , trained_on : dict = None):
    """
    save_model writes a model to a JSON file, whole or not at all, with what it was trained on
    (data range, horizon, zone) so a saved model can always be traced.
    It returns the path.

    Inputs:
    model : dict - a learned model
    path : str - where to write it
    trained_on : dict - description of the training data and settings
    """

    out = dict(model)
    out["trained_on"] = trained_on or {}
    os.makedirs(os.path.dirname(path) , exist_ok = True)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp , "w" , encoding = "utf-8") as f:
            json.dump(out , f , indent = 2)
        os.replace(tmp , path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path
