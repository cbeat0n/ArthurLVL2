### Arthur.py is a modular and agnostic framework that utilizes microwebframeworks
### and Python / Javascript for End to End turnkey hedge fund systems.
### The main differentiating factor at MVP is pre compliance, logging, speed to execution,
### accuracy of execution, and modularity / agnosticity so evolution
### is legal but outpaces regulation.

## External Dependencies
from dotenv import load_dotenv # Env hides API Key
import requests # Library for handling HTTP requests
import signal # Library for Server signals
from datetime import datetime , timezone # Library for time objects
from flask import Flask, render_template, request # Library for common MWF
from flask import jsonify , redirect, url_for # Library for common MWF
import pandas as pd # Library for data manipulation
import time # Library for time
import os , glob # Library for operating system communication
from threading import Event , Lock # Library for flagging black swans and guarding startup
from concurrent.futures import ThreadPoolExecutor # Threading for MVP, Multiprocessing for later

## Internal Dependencies
from utils.extract_quantitative import fetch_candles
from utils.transform_cleaned import convert_to_tabular , clean_tabular
from utils.load_cleaned import load_cleaned_data
from utils.transform_signal import decisionrule , transform_to_signal
from utils.load_signal import load_signal_matrix
from utils.operators import DataVendor , Platform , OandaTrading ,ArthurConfig, Arthur
from utils.operators import RiskManager , Trader
from utils.operators import OperationalState
from utils.granularity import horizon_seconds , check_granularity
from utils.scheduler import CycleScheduler

## Load environment variables from the .env file
load_dotenv()

## Initialize Flask App and globally hardcoded parameters.
ArthurFlask = Flask(__name__)
GLOBAL_STATE = {"Arthur" : None , "Scheduler" : None} # Arthur, its scheduler and state start at None
START_LOCK = Lock() # one startup at a time
STOP_WHEN_CLOSED = True # True: stop the server when CLOSED. False: stay up and wait for the next cycle
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data")
REGISTRY = {"vendor" : None , "platform" : os.getenv("OANDADEMOAPIKEY")} # Can keep .env separate
VENDOR = DataVendor() # Blank because MVP uses Oanda


@ArthurFlask.route("/", methods=["GET"])
def home():
    """
    home sets up a simple frontend intake page where the user is able to input parameters
    to modify Arthur. It is MVP friendly and scalable.
    It returns a rendered template.

    Inputs:
    None
    """
    
    return render_template("homepage.html")


def scheduled_cycle():
    """
    scheduled_cycle is what the scheduler runs once per horizon. It returns False to stop the
    scheduler, anything else to continue.

    Inputs:
    None
    """

    arthur = GLOBAL_STATE["Arthur"]
    state = arthur.run_once()
    GLOBAL_STATE["State"] = state

    if state == OperationalState.CLOSED and STOP_WHEN_CLOSED:
        arthur.shutdown() # Stops thread pooling
        print("Trading out of bounds for current operations.")
        requests.post("http://127.0.0.1:5000/shutdownserver")
        return False
    return True


@ArthurFlask.route("/accounting", methods = ["GET","POST"])
def enter_vault():
    """
    enter_vault starts Arthur and its scheduler on the first visit, then shows the dynamic
    accounting dashboard. Trading runs on the scheduler's own thread, once per horizon, so it
    no longer depends on this page being open or reloading.
    It returns a rendered template.

    Inputs:
    None
    """
    
    global GLOBAL_STATE , BASE_DIR , VENDOR , REGISTRY

    with START_LOCK: # two first visits at once must not start two bots

        if GLOBAL_STATE["Arthur"] is None:

            instrument = request.args.get("currencyPair", "USD_JPY")
            intake = request.args.get("intake", "S5")
            horizon = request.args.get("horizon", "M1Q0")
            zone = int(request.args.get("zone", "10"))
            managers = int(request.args.get("Managers", 1))
            traders = int(request.args.get("traders", 1))
            mode = request.args.get("mode", "Demo")

            ## Refuse a bad pairing now, not inside the scheduler thread every cycle
            try:
                check_granularity(intake, horizon)
            except ValueError as e:
                return str(e), 400

            ## Local variable to change it without scope violation
            platform = OandaTrading(REGISTRY["platform"])

            config = ArthurConfig(
                base_dir=BASE_DIR,
                mode=mode,
                vendor=VENDOR,
                platform=platform,
                registry=REGISTRY,
                instrument=instrument,
                intake=intake,
                horizon=horizon,
                zone=zone,
                threadpool=ThreadPoolExecutor(max_workers = managers*traders), # Static, dynamic later
                state=OperationalState.OPEN
            )
            
            ## Make Platform & Arthur Instance
            platform.set_config(config)
            platform.set_instrument(instrument) # Hardcoded instrument
            GLOBAL_STATE["Arthur"] = Arthur(cfg = config , teamstructure = {"managers" : managers , "traders" : traders})
            GLOBAL_STATE["State"] = config.operationalstate

            ## One cycle per horizon, on the horizon boundary, until the process ends
            GLOBAL_STATE["Scheduler"] = CycleScheduler(period_s = horizon_seconds(horizon), run_cycle = scheduled_cycle)
            GLOBAL_STATE["Scheduler"].start()

    arthur = GLOBAL_STATE["Arthur"]

    ## 4 way switch, not manual in hand switch
    if request.method == "POST":
        value = request.form.get("autoSwitch")
        if value == "AUTOMATED":
            arthur.config.holdingbutton = None
        elif value in OperationalState.__members__:
            arthur.config.holdingbutton = OperationalState[value]

    ## Accounting Information, plus how the scheduler is doing
    account_summary = arthur._accounting()["summary"]
    account_summary = {**account_summary ,
                       "Bot state" : arthur.config.operationalstate.value ,
                       **GLOBAL_STATE["Scheduler"].status()}

    return render_template("vault_storage.html", 
                           accountSummary = account_summary,
                           signalFile = "",
                           instrument = arthur.config.instrument,
                           refreshRate = arthur.config.horizon,
                           zone = arthur.config.zone)

def latest_signal_records(instrument):
    """Newest signal file for the instrument as a list of records, or None."""
    if instrument not in {"USD_JPY", "USD_CHF", "USD_MXN", "EUR_USD"}:
        return None
    files = glob.glob(os.path.join(BASE_DIR, "signal", "*", f"signal_data_{instrument}_*.csv"))
    if not files:
        return None
    df = pd.read_csv(max(files, key=os.path.getmtime))
    df = df.replace([float("inf"), float("-inf")], float("nan"))   # JSON has no Infinity
    return df.astype(object).where(df.notna(), None).to_dict("records")

@ArthurFlask.route("/visualize/", methods=["GET"])
def visualize_market():
    instrument = request.args.get("instrument", "")
    records = latest_signal_records(instrument)
    if records is None:
        return "No signal data yet or unknown instrument", 404
    return render_template("market_view.html", signalMatrix=records, instrument=instrument,
                           refreshRate=request.args.get("horizon", "M1Q0"),
                           zone=request.args.get("zone"))

@ArthurFlask.route("/visualize/data", methods=["GET"])
def visualize_data():
    records = latest_signal_records(request.args.get("instrument", ""))
    if records is None:
        return jsonify({"error": "no signal data"}), 404
    return jsonify(records)


@ArthurFlask.route("/shutdownserver", methods=["POST"])
def shut_down_server():
    """
    shut_down_server shuts down the Waitress server by sending SIGTERM to the current process.
    It returns nothing.

    Inputs:
    None
    """

    print("Server shutting down...")
    pid = os.getpid()
    os.kill(pid, signal.SIGTERM)   # Graceful termination signal




## Werkzeug for Development (slower, construction, setting up skeleton into chassis)
if __name__ == "__main__":
    ArthurFlask.run(debug=False)

## Waitress for Production -> Move to Gunicorn and Docker for full multiple computers
## When basis is ready, Waitress is stateless and spits out every last incorrect assertion.
## if __name__ == "__main__":
##     from waitress import serve
##     #serve(ArthurFlask, host="0.0.0.0", port=5000) # Waitress on non local server, closer to VPS
##     serve(ArthurFlask, host="127.0.0.1", port=5000)


