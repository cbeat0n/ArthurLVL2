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


# Initialize Flask App and needed parameters.
ArthurFlask = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# REMAKE GET/POST FOR AUTO TRADING LATER OR ADD /trade ROUTES FOR AUTO TRADING, DEFER
# Home Route
@ArthurFlask.route("/", methods=["GET"])
def home():
    return render_template("homepage.html")


@ArthurFlask.route("/accounting", methods = ["GET"])
def enter_vault():
    # homepage opens accounting. accounting starts up the Arthur instance.
    # accounting trades and sends information to viz market. Done.

    #

    return render_template("vault_storage.html")

@ArthurFlask.route("/visualize/<instrument>", methods=["GET"])
def visualize_market(instrument):
    # render template needs signal , instrument , and refresh rate.
    

    return render_template("market_view.html",
                           signalMatrix=signal_dict,
                           instrument=instrument,
                           granularity=intake)
    
if __name__ == "__main__":
    ArthurFlask.run(debug=True)


