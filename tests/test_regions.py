### test_regions.py checks the Theta x Mu region lookup that decisionrule uses and the research learner
### that builds it. No network, no key and no real data are used.

import json

import numpy as np
import pandas as pd
import pytest

import utils.regions as rt
import utils.transform_signal as ts
import research.regions as rg

MODEL = {"theta_cuts" : [5.0] , "mu_cuts" : [-0.5 , 0.5] , "actions" : [0 , 2 , 1 , 2 , 2 , 2]}
# Theta below 5: Mu low BUY, mid HOLD, high SELL. Theta above 5: all HOLD.


def test_region_lookup_places_each_pair():
    assert rt.region_id(1.0 , -1.0 , MODEL) == 0
    assert rt.region_id(1.0 , 0.0 , MODEL) == 1
    assert rt.region_id(1.0 , 2.0 , MODEL) == 2
    assert rt.region_id(9.0 , -1.0 , MODEL) == 3
    assert rt.region_action(1.0 , -1.0 , MODEL) == rt.BUY
    assert rt.region_action(1.0 , 2.0 , MODEL) == rt.SELL
    assert rt.region_action(9.0 , 2.0 , MODEL) == rt.HOLD


def test_values_that_cannot_be_placed_are_hold():
    for t , m in [(np.nan , 0.0) , (1.0 , np.nan) , (rt.SENTINEL , 0.0) , (1.0 , rt.SENTINEL) , (None , 0.0) , ("x" , 0.0) , (np.inf , 0.0)]:
        assert rt.region_action(t , m , MODEL) == rt.HOLD


def test_missing_or_broken_model_file_is_hold(tmp_path , monkeypatch):
    monkeypatch.setattr(rt , "MODEL_PATH" , str(tmp_path / "none.json"))
    assert rt.load_model() is None and rt.region_action(1.0 , -1.0) == rt.HOLD
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"theta_cuts" : [3 , 1] , "mu_cuts" : [] , "actions" : [0 , 0 , 0]}))
    assert rt.load_model(str(bad)) is None
    bad.write_text("not json")
    assert rt.load_model(str(bad)) is None


def test_model_file_is_read_again_when_it_changes(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps(MODEL))
    assert rt.load_model(str(path))["actions"][0] == 0
    path.write_text(json.dumps({**MODEL , "actions" : [1 , 2 , 1 , 2 , 2 , 2]}))
    import os
    os.utime(path , (path.stat().st_atime + 5 , path.stat().st_mtime + 5))
    assert rt.load_model(str(path))["actions"][0] == 1


def row(theta = 1.0 , mu = -1.0):
    r = {c : 1.0 for c in ts.SIGNAL_COLS}
    r.update({"Theta" : theta , "Mu" : mu})
    return pd.Series(r)


def test_decisionrule_default_never_trades(monkeypatch):
    assert ts.DECISION_MODE == "hold"
    monkeypatch.setattr(rt , "load_model" , lambda path = None : MODEL)
    assert ts.decisionrule(row()) == 2


def test_decisionrule_regions_mode_reads_the_map(monkeypatch):
    monkeypatch.setattr(ts , "DECISION_MODE" , "regions")
    monkeypatch.setattr(rt , "load_model" , lambda path = None : MODEL)
    assert ts.decisionrule(row(1.0 , -1.0)) == 0
    assert ts.decisionrule(row(1.0 , 2.0)) == 1
    assert ts.decisionrule(row(9.0 , 0.0)) == 2
    warming = row(); warming["B"] = np.nan
    assert ts.decisionrule(warming) == 2 # a row still warming up is HOLD in any mode


def planted(n = 4000 , edge = 4.0 , noise = 6.0 , seed = 0):
    """Mu below -1 gives long trades a positive result and Mu above 1 gives short trades one."""
    rng = np.random.default_rng(seed)
    theta , mu = rng.uniform(0 , 10 , n) , rng.normal(0 , 1.5 , n)
    drift = np.where(mu < -1 , edge , np.where(mu > 1 , -edge , 0.0))
    long = drift + rng.normal(0 , noise , n)
    return pd.DataFrame({"Theta" : theta , "Mu" : mu , "long_pips" : long , "short_pips" : -long})


def test_learner_finds_a_planted_edge():
    model = rg.learn_regions(planted() , shape = (1 , 4) , spread = 1.0)
    acts = apply = rg.apply_regions(pd.DataFrame({"Theta" : [5.0 , 5.0] , "Mu" : [-3.0 , 3.0]}) , model)
    assert list(acts) == [rt.BUY , rt.SELL]


def test_learner_finds_nothing_in_noise():
    rng = np.random.default_rng(1)
    n = 4000
    df = pd.DataFrame({"Theta" : rng.uniform(0 , 10 , n) , "Mu" : rng.normal(0 , 1 , n) ,
                       "long_pips" : rng.normal(0 , 6 , n)})
    df["short_pips"] = -df["long_pips"]
    model = rg.learn_regions(df , shape = (2 , 3) , spread = 1.6)
    assert set(model["actions"]) == {rt.HOLD}


def test_thin_regions_do_not_trade():
    df = planted(n = 90)
    model = rg.learn_regions(df , shape = (1 , 3) , spread = 0.0 , min_n = 150)
    assert set(model["actions"]) == {rt.HOLD}


def test_spread_can_turn_an_edge_into_hold():
    cheap = rg.learn_regions(planted() , shape = (1 , 4) , spread = 0.5)
    costly = rg.learn_regions(planted() , shape = (1 , 4) , spread = 50.0)
    assert rt.BUY in cheap["actions"] and set(costly["actions"]) == {rt.HOLD}


def test_saved_model_loads(tmp_path):
    model = rg.learn_regions(planted() , shape = (2 , 2) , spread = 1.0)
    path = rg.save_model(model , str(tmp_path / "models" / "m.json") , {"horizon_min" : 30})
    loaded = rt.load_model(path)
    assert loaded["actions"] == model["actions"] and loaded["trained_on"]["horizon_min"] == 30
