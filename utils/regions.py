### regions.py is a modular and agnostic framework that turns Theta and Mu into a trading action.
### The Theta x Mu plane is cut into a small grid of regions. Each region maps to 0 BUY , 1 SELL or 2 HOLD.
### The map is learned offline by research/regions.py and saved in models/model_params.json, so
### retraining never means editing code. Anything unknown, missing or not a number is HOLD.

## External Dependencies
import json , os # Libraries for the saved model file
import numpy as np # Library for numerical work

BUY , SELL , HOLD = 0 , 1 , 2 # same codes as decisionrule
SENTINEL = 0.00000000314 # transform_signal fills unknown values with this
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) , "models" , "model_params.json")

_cache = {} # path -> (mtime, model) so the file is read again only when it changes


def load_model(path : str = None):
    """
    load_model reads the saved region model. The file is read again only when it has changed.
    It returns the model dictionary, or None if there is no file or it can't be read, so
    callers fall back to HOLD.

    Inputs:
    path : str - path to the JSON model, defaults to models/model_params.json
    """

    path = path or MODEL_PATH
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None

    hit = _cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1]

    try:
        with open(path , "r" , encoding = "utf-8") as f:
            model = json.load(f)
        check_model(model)
    except (OSError , ValueError , KeyError , TypeError):
        return None

    _cache[path] = (mtime , model)
    return model


def check_model(model : dict):
    """
    check_model raises ValueError if a model is not usable: cut points must be sorted lists and
    every region must have an action in {0 , 1 , 2}.
    It returns nothing.

    Inputs:
    model : dict - the loaded model
    """

    for key in ("theta_cuts" , "mu_cuts"):
        cuts = model[key]
        if not isinstance(cuts , list) or any(not np.isfinite(c) for c in cuts) or cuts != sorted(cuts):
            raise ValueError(f"{key} must be a sorted list of finite numbers")

    n_regions = (len(model["theta_cuts"]) + 1) * (len(model["mu_cuts"]) + 1)
    actions = model["actions"]
    if len(actions) != n_regions or any(a not in (BUY , SELL , HOLD) for a in actions):
        raise ValueError(f"actions must hold {n_regions} values from {{0 , 1 , 2}}")


def region_id(theta : float , mu : float , model : dict):
    """
    region_id finds which region a Theta and Mu pair falls in. Regions are numbered with Theta bins
    as rows and Mu bins as columns: id = theta_bin * (number of Mu bins) + mu_bin.
    It returns an integer, or None if either value is missing, not finite or the sentinel.

    Inputs:
    theta : float - Theta in degrees
    mu : float - Mu
    model : dict - the loaded model
    """

    try:
        theta , mu = float(theta) , float(mu)
    except (TypeError , ValueError):
        return None

    if not (np.isfinite(theta) and np.isfinite(mu)):
        return None
    if abs(theta - SENTINEL) < 1e-12 or abs(mu - SENTINEL) < 1e-12:
        return None

    t_bin = int(np.searchsorted(model["theta_cuts"] , theta , side = "right"))
    m_bin = int(np.searchsorted(model["mu_cuts"] , mu , side = "right"))
    return t_bin * (len(model["mu_cuts"]) + 1) + m_bin


def region_action(theta : float , mu : float , model : dict = None):
    """
    region_action is the decision for one bar: the action of the region that Theta and Mu fall in.
    It returns 0 BUY , 1 SELL or 2 HOLD. HOLD is the answer whenever there is no model or the values
    can't be placed, so a missing file or a warming-up row never trades.

    Inputs:
    theta : float - Theta in degrees
    mu : float - Mu
    model : dict - a loaded model, defaults to the saved one
    """

    model = model if model is not None else load_model()
    if model is None:
        return HOLD

    rid = region_id(theta , mu , model)
    if rid is None:
        return HOLD
    return int(model["actions"][rid])
