### test_pull_history.py checks the history puller with a stand-in for OANDA, so no network or key is used.

import os

import pandas as pd
import pytest

import research.pull_history as ph


def candle(ts , complete = True , price = 150.0):
    q = lambda p: {"o" : f"{p:.3f}" , "h" : f"{p + .01:.3f}" , "l" : f"{p - .01:.3f}" , "c" : f"{p:.3f}"}
    return {"time" : ts , "complete" : complete , "volume" : 7 , "mid" : q(price) , "bid" : q(price - .0185) , "ask" : q(price + .0185)}


class FakeResponse:
    status_code = 200
    def __init__(self , body): self._b = body
    def json(self): return self._b


class FakeHttp:
    """Answers each from/to request with the candles of a fixed list that fall inside it."""
    def __init__(self , candles): self.candles , self.calls = candles , []
    def get(self , url , params):
        self.calls.append(params)
        lo , hi = params["from"] , params["to"]
        inside = [c for c in self.candles if lo <= c["time"][:19] + "Z" < hi]
        return FakeResponse({"candles" : inside})


@pytest.fixture(autouse = True)
def fast(monkeypatch , tmp_path):
    monkeypatch.setattr(ph , "THROTTLE_S" , 0)
    monkeypatch.setattr(ph , "OUT_ROOT" , str(tmp_path))


def test_candles_to_frame_keeps_complete_sorts_and_dedupes():
    df = ph.candles_to_frame([candle("2026-01-05T00:02:00.000000000Z") , candle("2026-01-05T00:01:00.000000000Z") ,
                              candle("2026-01-05T00:01:00.000000000Z") , candle("2026-01-05T00:03:00.000000000Z" , complete = False)])
    assert df["timestamp"].tolist() == ["2026-01-05 00:01:00+00:00" , "2026-01-05 00:02:00+00:00"]
    assert list(df.columns) == ph.COLUMNS
    assert (df["bid_c"] < df["ask_c"]).all()


def test_month_ranges_clips_to_the_range():
    r = ph.month_ranges("2026-01-15" , "2026-03-10")
    assert [x[0] for x in r] == ["2026-01" , "2026-02" , "2026-03"]
    assert r[0][1].day == 15 and r[-1][2].strftime("%Y-%m-%d") == "2026-03-11"
    with pytest.raises(ValueError):
        ph.month_ranges("2026-03-10" , "2026-01-01")


def test_pull_writes_one_file_per_month_and_skips_finished_months():
    cs = [candle("2026-01-05T10:00:00.000000000Z") , candle("2026-02-03T10:00:00.000000000Z")]
    http = FakeHttp(cs)
    paths = ph.pull_history("USD_JPY" , "M1" , "2026-01-01" , "2026-02-28" , http = http)
    assert len(paths) == 2 and all(os.path.exists(p) for p in paths)
    assert len(pd.read_csv(paths[0])) == 1 and len(pd.read_csv(paths[1])) == 1
    n = len(http.calls)
    ph.pull_history("USD_JPY" , "M1" , "2026-01-01" , "2026-02-28" , http = http)
    assert len(http.calls) == n # second run fetched nothing


def test_requests_stay_under_the_chunk_size():
    http = FakeHttp([])
    ph.pull_history("USD_JPY" , "M1" , "2026-01-01" , "2026-01-31" , http = http)
    assert len(http.calls) == 11 # 31 days in 3 day chunks
    assert all(c["price"] == "MBA" and c["granularity"] == "M1" for c in http.calls)


def test_s5_requests_are_six_hours_each():
    http = FakeHttp([])
    ph.pull_history("USD_JPY" , "S5" , "2026-01-01" , "2026-01-02" , http = http)
    assert len(http.calls) == 8 # 2 days in 6 hour chunks: 4320 S5 candles each, under 5000
    assert all(c["granularity"] == "S5" for c in http.calls)


def test_granularity_without_a_chunk_size_is_refused():
    with pytest.raises(ValueError):
        ph.pull_history("USD_JPY" , "H1" , "2026-01-01" , "2026-01-02" , http = FakeHttp([]))


def test_unsupported_instrument_is_refused():
    with pytest.raises(ValueError):
        ph.pull_history("GBP_JPY" , "M1" , "2026-01-01" , "2026-01-02" , http = FakeHttp([]))
