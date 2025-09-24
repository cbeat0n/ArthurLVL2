### Arthur.py is a modular and agnostic framework that utilizes microwebframeworks
### and Python / Javascript for End to End turnkey hedge fund systems.
### The main differentiating factor at MVP is pre compliance, logging, speed to execution,
### accuracy of execution, and modularity / agnosticity so evolution
### is legal but outpaces regulation.

## External Dependencies
from dotenv import load_dotenv # Env hides API Key
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY = {"vendor" : None , "platform" : os.getenv("OANDADEMOAPIKEY")} # Can keep .env separate
VENDOR = DataVendor() # Blank because MVP uses Oanda
PLATFORM = OandaTrading(REGISTRY['platform'])
EVENT = Event()




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

    instrument = request.args.get("currencyPair", "USD_JPY")
    intake = request.args.get("intake", "S5")
    horizon = request.args.get("horizon", "M1")
    zone = request.args.get("zone", "10")
    managers = int(request.args.get("Managers", 1))
    traders = int(request.args.get("traders", 1))
    mode = request.args.get("mode", "Demo")

    config = ArthurConfig(
        base_dir=BASE_DIR,
        mode=mode,
        vendor=VENDOR,
        platform=PLATFORM,
        registry=REGISTRY,
        instrument=instrument,
        intake=intake,
        horizon=horizon,
        zone=zone,
        threadpool=ThreadPoolExecutor(max_workers = managers*traders), # Static, dynamic later
        event=EVENT
    )
    ArthurInstance = Arthur(cfg = config , teamstructure = {"managers" : managers , "traders" : traders})

    # logic that sets up accounting page params , runs the trades , and sets up params for market _ view.
    account_summary = ArthurInstance._accounting()
    ignition_center = ArthurInstance._ignition_A()
    signal_matrix , signal_filename = ArthurInstance._acquire_market()
    
    if ignition_center == "OPEN":
        ArthurInstance._market_cycle(marketfile=signal_filename)
    else:
        print("Trading out of bounds for current business operations.")
        print("Check operational integrity for reasoning.")


    return render_template("vault_storage.html", 
                           accountSummary = account_summary,
                           signalFile = signal_filename,
                           instrument = instrument,
                           refreshRate = horizon)

@ArthurFlask.route("/visualize/", methods=["GET"])
def visualize_market():
    """
    visualize_market sets up a dynamic UI so that the guardian of Arthur can see the data
    rendered by proprietary algorithm using proprietary UI.
    It returns a rendered template.

    Inputs:
    None
    """
    
    filepath = request.get.args("signalFile")
    ui_signal_matrix = pd.read_csv(filepath).to_dict("records")
    ui_instrument = request.get.args("instrument")
    ui_horizon = request.get.args("horizon")

    return render_template("market_view.html",
                           signalMatrix=ui_signal_matrix,
                           instrument=ui_instrument,
                           refreshRate=ui_horizon)
    

## Debug = True for BASH logging
if __name__ == "__main__":
    ArthurFlask.run(debug=True)


