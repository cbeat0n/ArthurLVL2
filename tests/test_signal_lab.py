### test_signal_lab.py checks the research harness: the sealed months stay sealed, outcome labels are what
### they say, and the walk-forward never trains on a month it is about to be scored on.

import os

import numpy as np
import pandas as pd
import pytest

import research.signal_lab as lab
from research.pull_history import write_atomic, COLUMNS


def make_history(root , months = ("2026-06" , "2026-07" , "2026-08") , per = 50):
    folder = os.path.join(root , "USD_JPY_S5")
    for m in months:
        ts = pd.date_range(f"{m}-03 10:00" , periods = per , freq = "5s" , tz = "UTC")
        df = pd.DataFrame({c : [150.0] * per for c in COLUMNS})
        df["timestamp"] = ts.strftime("%Y-%m-%d %H:%M:%S+00:00")
        df["volume"] = 10
        write_atomic(df , os.path.join(folder , f"USD_JPY_S5_{m}.csv.gz"))


def test_sealed_months_are_not_loaded_unless_asked(tmp_path):
    make_history(str(tmp_path))
    sealed = lab.load_history("USD_JPY" , "S5" , str(tmp_path))
    assert sealed.index.max() < lab.HOLDOUT_START
    assert lab.load_history("USD_JPY" , "S5" , str(tmp_path) , allow_holdout = True).index.max() >= lab.HOLDOUT_START


def test_missing_history_says_how_to_fix_it(tmp_path):
    with pytest.raises(FileNotFoundError , match = "pull_history"):
        lab.load_history("USD_JPY" , "S5" , str(tmp_path))


def bars_from(highs , lows , closes):
    n = len(closes)
    return pd.DataFrame({"timestamp" : pd.date_range("2026-01-05" , periods = n , freq = "30min" , tz = "UTC") ,
                         "open" : closes , "high" : highs , "low" : lows , "close" : closes , "volume" : [1] * n})


def label(highs , lows , closes , target = 10 , stop = 5 , touch = 3):
    bars = bars_from(highs , lows , closes)
    sig = pd.DataFrame({"i" : [0] , "timestamp" : [bars.timestamp[0]] , "Theta" : [1.0] , "Mu" : [0.0] , "Margin" : [0.0]})
    out = lab.add_outcomes(sig , bars , "USD_JPY" , [(target , stop)] , touch_bars = touch)
    return out.iloc[0]


def test_target_before_stop_pays_the_target():
    # entry 150.00. bar 1 reaches +12 pips and only -2: long hits the 10 pip target, short hits its 5 pip stop.
    r = label([150.0 , 150.12 , 150.0 , 150.0] , [150.0 , 149.98 , 150.0 , 150.0] , [150.0 , 150.0 , 150.0 , 150.0])
    assert r["long_T10_S5"] == 10 and r["short_T10_S5"] == -5


def test_stop_first_and_both_in_one_bar_count_as_the_stop():
    r = label([150.0 , 150.01 , 150.0 , 150.0] , [150.0 , 149.94 , 150.0 , 150.0] , [150.0] * 4)
    assert r["long_T10_S5"] == -5
    both = label([150.0 , 150.12 , 150.0 , 150.0] , [150.0 , 149.94 , 150.0 , 150.0] , [150.0] * 4)
    assert both["long_T10_S5"] == -5 and both["short_T10_S5"] == -5


def test_no_touch_exits_at_the_last_close_and_runs_past_data_have_no_label():
    r = label([150.0 , 150.02 , 150.03 , 150.03] , [150.0 , 149.99 , 150.0 , 150.0] , [150.0 , 150.01 , 150.02 , 150.03])
    assert r["long_T10_S5"] == pytest.approx(3.0) and r["short_T10_S5"] == pytest.approx(-3.0)
    short_data = label([150.0 , 150.0] , [150.0 , 150.0] , [150.0 , 150.0])
    assert np.isnan(short_data["long_T10_S5"])


def test_next_bar_label_is_the_following_close():
    bars = bars_from([150.1] * 3 , [149.9] * 3 , [150.00 , 150.07 , 149.99])
    sig = pd.DataFrame({"i" : [0 , 1] , "timestamp" : bars.timestamp[:2] , "Theta" : [1.0] * 2 , "Mu" : [0.0] * 2 , "Margin" : [0.0] * 2})
    out = lab.add_outcomes(sig , bars)
    assert out.long_pips.tolist() == pytest.approx([7.0 , -8.0]) and out.short_pips.tolist() == pytest.approx([-7.0 , 8.0])


def test_score_counts_spread_once_per_trade():
    df = pd.DataFrame({"long_pips" : [5.0 , 1.0 , 2.0] , "short_pips" : [-5.0 , -3.0 , -2.0]})
    res = lab.score(df , np.array([0 , 1 , 2]) , spread = 1.6) # buy wins 5, sell loses 3, third row is HOLD
    assert res["trades"] == 2 and res["hit"] == pytest.approx(0.5)
    assert res["gross"] == pytest.approx(1.0) and res["net"] == pytest.approx(-0.6) and res["net_total"] == pytest.approx(-1.2)


def monthly_frame(months = 10 , per = 60 , seed = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(months):
        ts = pd.date_range(f"2026-{k + 1:02d}-05" , periods = per , freq = "1h" , tz = "UTC")
        mu = rng.normal(0 , 1.5 , per)
        long = np.where(mu < -1 , 4.0 , 0.0) + rng.normal(0 , 5 , per)
        rows.append(pd.DataFrame({"timestamp" : ts , "Theta" : rng.uniform(0 , 10 , per) , "Mu" : mu ,
                                  "long_pips" : long , "short_pips" : -long , "ret0" : rng.normal(0 , 1 , per)}))
    return pd.concat(rows , ignore_index = True)


def test_walk_forward_never_trains_on_the_gap_or_the_test_month(monkeypatch):
    seen = []
    real = lab.learn_regions
    def spy(train , **kw):
        seen.append(sorted(train.index))
        return real(train , **kw)
    monkeypatch.setattr(lab , "learn_regions" , spy)

    df = monthly_frame()
    monthly , pooled = lab.walk_forward(df , shape = (1 , 3) , train_months = 4 , gap_months = 1 , min_train_months = 3 , min_n = 10)
    assert len(monthly) > 0 and "regions" in pooled and "always_buy" in pooled and "fade_last_bar" in pooled

    month = lab.month_index(df["timestamp"])
    # the latest training row used in each fit is at least gap_months + 1 months before the month being tested
    for (_ , r) , used in zip(monthly.iterrows() , seen):
        y , m = map(int , r["month"].split("-"))
        T = y * 12 + m
        assert month.loc[used].max() <= T - 2


def test_walk_forward_with_too_little_history_returns_nothing():
    monthly , pooled = lab.walk_forward(monthly_frame(months = 3) , train_months = 4 , gap_months = 1 , min_train_months = 3)
    assert len(monthly) == 0 and pooled == {}


def test_trial_log_appends(tmp_path):
    path = str(tmp_path / "trials.csv")
    lab.log_trial({"horizon_min" : 30 , "zone" : 5} , {"regions" : {"trades" : 10 , "net" : -1.2345678}} , path)
    lab.log_trial({"horizon_min" : 45 , "zone" : 5} , {"regions" : {"trades" : 12 , "net" : 0.5}} , path)
    log = pd.read_csv(path)
    assert len(log) == 2 and log["set_horizon_min"].tolist() == [30 , 45] and log["regions_net"].iloc[0] == pytest.approx(-1.2346)
