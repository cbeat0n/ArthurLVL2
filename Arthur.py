### Arthur.py is a modular and agnostic framework that utilizes microwebframeworks
### and Python / Javascript for End to End turnkey hedge fund systems.
### The main differentiating factor at MVP is pre compliance, logging, speed to execution,
### accuracy of execution, and modularity / agnosticity so evolution
### is legal but outpaces regulation.

## External Dependencies
from dotenv import load_dotenv # Env hides API Key
import requests # Library for handling HTTP requests
from datetime import datetime , timezone # Library for time objects
from flask import Flask, render_template, request # Flask is common microwebframework
import pandas as pd # Library for data manipulation
import time # Library for time
import os # Library for operating system communication
from threading import Event # Library for flagging black swans
from concurrent.futures import ThreadPoolExecutor # Threading for MVP, Multiprocessing for later

## Internal Dependencies
from utils.extract_quantitative import fetch_candles
from utils.transform_cleaned import convert_to_tabular , clean_tabular
from utils.load_cleaned import load_cleaned_data
from utils.transform_signal import decisionrule , transform_to_signal
from utils.load_signal import load_signal_matrix
from utils.operators import DataVendor , Platform , OandaTrading ,ArthurConfig, Arthur
from utils.operators import RiskManager , Trader

## Load environment variables from the .env file
load_dotenv()

## Initialize Flask App and globally hardcoded parameters.
ArthurFlask = Flask(__name__)
GLOBAL_STATE = {"Arthur" : None , "State" : None} # Arthur and its corresponding state start at None
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data")
REGISTRY = {"vendor" : None , "platform" : os.getenv("OANDADEMOAPIKEY")} # Can keep .env separate
VENDOR = DataVendor() # Blank because MVP uses Oanda
EVENT = ... ## make enum for open closed holding




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


@ArthurFlask.route("/accounting", methods = ["GET"])
def enter_vault():
    """
    enter_vault sets up the dynamic accounting dashboard for current status
    on trading, and sends dynamically refreshed information to visualize route.
    It returns a rendered template.

    Inputs:
    None
    """
    
    global GLOBAL_STATE , BASE_DIR , VENDOR , REGISTRY


    if GLOBAL_STATE["Arthur"] == None:

        instrument = request.args.get("currencyPair", "USD_JPY")
        intake = request.args.get("intake", "S5")
        horizon = request.args.get("horizon", "M1")
        zone = int(request.args.get("zone", "10"))
        managers = int(request.args.get("Managers", 1))
        traders = int(request.args.get("traders", 1))
        mode = request.args.get("mode", "Demo")

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
            event=EVENT
        )
        
        ## Make Platform & Arthur Instance
        platform.set_config(config)
        platform.set_instrument(instrument) # Hardcoded instrument
        GLOBAL_STATE["Arthur"] = Arthur(cfg = config , teamstructure = {"managers" : managers , "traders" : traders})
        
    
    else:
        print(f"GLOBAL_STATE of Arthur INSTANTIATED as of {datetime.now(timezone.utc)} UTC")
        
    arthur = GLOBAL_STATE["Arthur"]

    ## Accounting Information
    account_summary = arthur._accounting()["summary"]

    ## Arthur Ignition
    ignition_center = arthur._ignition_A()

    ## Acquire Market information
    signal_filename = arthur._acquire_market()["location"]
    
    if ignition_center == "OPEN":
        arthur._market_cycle(marketfile=signal_filename)
    else:
        arthur.shutdown()
        print("Trading out of bounds for current business operations.")
        print("Check logs for information.")
        raise SystemExit(0)
        #requests.post("http://127.0.0.1:5000/shutdownserver")

    return render_template("vault_storage.html", 
                           accountSummary = account_summary,
                           signalFile = signal_filename,
                           instrument = arthur.config.instrument,
                           refreshRate = arthur.config.horizon,
                           zone = arthur.config.zone)

@ArthurFlask.route("/visualize/", methods=["GET"])
def visualize_market():
    """
    visualize_market sets up a dynamic UI so that the guardian of Arthur can see the data
    rendered by proprietary algorithm using proprietary UI.
    It returns a rendered template.

    Inputs:
    None
    """
    
    ## File of the signal for lightweight communication
    filepath = request.args.get("signalFile")

    ## JSON of the signal
    ui_signal_matrix = pd.read_csv(filepath).to_dict("records")

    ## Instrument and Refresh Rate as Horizon
    ui_instrument = request.args.get("instrument")
    ui_horizon = request.args.get("horizon")
    ui_zone = request.args.get("zone")

    return render_template("market_view.html",
                           signalMatrix=ui_signal_matrix,
                           instrument=ui_instrument,
                           refreshRate=ui_horizon,
                           zone = ui_zone)


@ArthurFlask.route("/shutdownserver", methods=["POST"])
def shut_down_server():
    """
    shut_down_server shuts down the Werkzeug in a mooth manner.
    It returns a string.

    Inputs:
    None
    """
    
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    
    func()
    
    return "Server shutting down..."  # Smooth shutdown

## Werkzeug for Development (slower, construction, setting up skeleton into chassis)
#if __name__ == "__main__":
#    ArthurFlask.run(debug=False)

## Waitress for Production -> Move to Gunicorn and Docker for full multiple computers
## When basis is ready, Waitress is stateless and spits out every last incorrect assertion.
if __name__ == "__main__":
    from waitress import serve
    #serve(ArthurFlask, host="0.0.0.0", port=5000) # Waitress on non local server, closer to VPS
    serve(ArthurFlask, host="127.0.0.1", port=5000)


