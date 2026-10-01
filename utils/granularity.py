### granularity.py is a modular and agnostic framework that converts intakes and
### horizons to seconds so any OANDA intake can be paired with any MXQN horizon.

## External Dependencies
import re # Library for RegEx

## OANDA intake granularities in seconds. W and M omitted, they have no fixed length.
INTAKE_SECONDS = {
    "S5": 5, "S10": 10, "S15": 15, "S30": 30,
    "M1": 60, "M2": 120, "M4": 240, "M5": 300, "M10": 600, "M15": 900, "M30": 1800,
    "H1": 3600, "H2": 7200, "H3": 10800, "H4": 14400, "H6": 21600, "H8": 28800, "H12": 43200,
    "D": 86400,
}

## Horizon of form MXQN, X minutes plus N quarter minutes, e.g. M5Q2 for M5.5
HORIZON_PATTERN = re.compile(r"^M(\d+)Q([0-3])$")


def intake_seconds(intake : str):
    """
    intake_seconds converts an OANDA intake granularity to seconds.
    It returns an integer.

    Inputs:
    intake : str - OANDA granularity e.g S5 , S30 , M1
    """

    if intake not in INTAKE_SECONDS:
        raise ValueError(f"Unknown intake '{intake}'. Expected one of {list(INTAKE_SECONDS)}.")
    return INTAKE_SECONDS[intake]


def horizon_seconds(horizon : str):
    """
    horizon_seconds converts an MXQN horizon to seconds, 60 seconds per minute and
    15 seconds per quarter minute.
    It returns an integer.

    Inputs:
    horizon : str - horizon of form MXQN e.g M1Q0 , M5Q2 , M10Q0
    """

    parts = HORIZON_PATTERN.match(horizon)
    if not parts:
        raise ValueError(f"Horizon '{horizon}' must be of form MXQN with N in 0-3, e.g. M1Q0 or M5Q2.")

    seconds = 60 * int(parts.group(1)) + 15 * int(parts.group(2))
    if seconds == 0:
        raise ValueError("Horizon must be longer than zero seconds.")
    return seconds


def check_granularity(intake : str , horizon : str):
    """
    check_granularity verifies a horizon is a whole number of intake candles, so every
    horizon sample is built from complete candles.
    It returns the number of intake candles per horizon sample as an integer.

    Inputs:
    intake : str - OANDA granularity e.g S5
    horizon : str - horizon of form MXQN e.g M5Q2
    """

    i_s = intake_seconds(intake)
    h_s = horizon_seconds(horizon)
    if h_s % i_s != 0:
        raise ValueError(f"Horizon {horizon} ({h_s}s) is not a whole number of {intake} candles ({i_s}s). "
                         f"Compatible intakes: {compatible_intakes(horizon)}")
    return h_s // i_s


def compatible_intakes(horizon : str):
    """
    compatible_intakes lists every intake that divides the horizon evenly.
    It returns a list of strings.

    Inputs:
    horizon : str - horizon of form MXQN e.g M5Q2
    """

    h_s = horizon_seconds(horizon)
    return [k for k, v in INTAKE_SECONDS.items() if h_s % v == 0]
