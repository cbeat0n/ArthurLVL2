### Arthur.py is is an an agnostic framework that serves as the main data pipeline,
### automation center, and engine for driving the financial Agentic AI.
### On this module there is the usage of ETLTL and later ETL-A and anything else added
### to maintain and grow the Agentic AI known as Arthur according to the personal agreement
### between the Parties.

# External Dependencies
from flask import Flask, render_template, request
import pandas as pd
import time
import os

# Internal Dependencies
from utils.extract_quantitative import fetch_candles
from utils.transform_cleaned import convert_to_tabular , clean_tabular
from utils.load_cleaned import get_today_folder , load_cleaned_data
from utils.transform_signal import acquire_proto_signal , decisionrule , transform_to_signal
from utils.load_signal import load_signal_matrix
from utils.operators import DataVendor , Platform , OandaTrading ,ArthurConfig, Arthur
from utils.operators import RiskManager , Trader


# Initialize Flask App and hardcoded parameters.
ArthurFlask = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENDOR = ...
PLATFORM = ...
REGISTRY = ...



# Home Route
@ArthurFlask.route("/", methods=["GET"])
def home():
    return render_template("homepage.html")


@ArthurFlask.route("/accounting", methods = ["GET"])
def enter_vault():
    # homepage opens accounting. accounting starts up the Arthur instance.
    #  accounting trades and sends information to viz market. Done.

    instrument = ... # these params use request.args and take from homepage
    intake = ...
    horizon = ...
    managers = ...
    teamsize = ...
    mode = ...

    config = ArthurConfig(mode = mode , vendor = VENDOR , platform = PLATFORM  , instrument = instrument,
                 intake = intake, horizon  = horizon)
    ArthurInstance = Arthur(cfg = config , managers = managers ,teamsize=teamsize, base_dir= BASE_DIR)
    ArthurInstance.set_registry(REGISTRY)


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
    filepath = request.get.args("signalFile")
    ui_signal_matrix = pd.read_csv(filepath).to_dict("records")
    ui_instrument = request.get.args("instrument")
    ui_horizon = request.get.args("horizon")

    return render_template("market_view.html",
                           signalMatrix=ui_signal_matrix,
                           instrument=ui_instrument,
                           refreshRate=ui_horizon)
    
if __name__ == "__main__":
    ArthurFlask.run(debug=True)


