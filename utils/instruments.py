### instruments.py is a modular and agnostic framework that holds what Arthur knows
### about each instrument it may trade. Live trading, demo and backtesting all read it from here.

## Supported instruments and their pip sizes. A pip is 0.01 for JPY quotes and 0.0001 for the rest here.
## OANDA names pairs BASE_QUOTE, so "USD per EUR" is EUR_USD. There is no USD_EUR.
PIP = {"USD_JPY" : 0.01 , "USD_CHF" : 0.0001 , "USD_MXN" : 0.0001 , "EUR_USD" : 0.0001}

SUPPORTED = tuple(PIP) # instruments Arthur accepts


def pip_size(symbol = "USD_JPY"):
    """
    pip_size defines the size of a pip for an FX pair, as a price difference.
    It raises ValueError for an instrument it doesn't know, so a new pair can't silently
    be measured with the wrong pip. Add it to PIP (after checking OANDA's pipLocation) first.

    Inputs:
    symbol : str - instrument symbol, e.g. USD_JPY
    """

    if symbol in PIP:
        return PIP[symbol]
    if isinstance(symbol , str) and symbol.endswith("_JPY"):
        return 0.01 # any JPY quote
    raise ValueError(f"Unknown instrument {symbol!r}: add its pip size to utils/instruments.py (supported: {', '.join(SUPPORTED)})")
