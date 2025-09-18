### Arthur.py is is an an agnostic framework that serves as the main data pipeline,
### automation center, and engine for driving the financial Agentic AI.
### On this module there is the usage of ETLTL and later ETL-A and anything else added
### to maintain and grow the Agentic AI known as Arthur according to the personal agreement
### between the Parties.

# External Dependencies
from flask import Flask, render_template, request
import pandas as pd
import os

# Internal Dependencies
from utils.extract_quantitative import fetch_candles
from utils.transform_cleaned import convert_to_tabular , clean_tabular
from utils.load_cleaned import get_today_folder , load_cleaned_data
from utils.transform_signal import acquire_proto_signal , decisionrule , transform_to_signal
from utils.load_signal import load_signal_matrix


# Initialize Flask App and needed parameters.
Arthur = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# REMAKE GET/POST FOR AUTO TRADING LATER OR ADD /trade ROUTES FOR AUTO TRADING, DEFER
# Home Route
@Arthur.route("/", methods=["GET"])
def home():

    # Step 0: Grab information on FX market from Text Input
    market = request.args.get("currencyPair", default = "USD_JPY")

    # Debug statement won't show Arthur if no market selected or received or JS Error.
    if not market:
        raise ValueError("No market selected or received.")
    # For now assume that USD/JPY hardcoded. Change later. Get homepage.html working.

    # Step 1: Extract raw candle data
    candles, instrument, granularity = fetch_candles(instrument = market)

    # Step 2: Convert and clean
    tabular_data = convert_to_tabular(candles)
    cleaned_data, safe_timestamp = clean_tabular(tabular_data)

    # Step 3: Load Cleaned Data
    data_folder = get_today_folder(os.path.join(BASE_DIR, "data"))
    
    filename = load_cleaned_data(data_folder, cleaned_data, instrument, safe_timestamp)
   
    # Step 4: Generate Signal Matrix
    proto_signal = acquire_proto_signal(filename)
    signal_matrix = transform_to_signal(proto_signal)

    # Step 5: Load Signal Matrix
    signal_filename = load_signal_matrix(data_folder , signal_matrix, instrument, safe_timestamp)
 

    # Step 6: Prepare for Loading Statistics
    latest_signal = signal_matrix.iloc[-1]
    b_val = round(latest_signal["B"], 3)
    s_val = round(latest_signal["S"], 3)
    theta = round(latest_signal["theta"], 2)
    action = int(latest_signal["ActionHat"])
    action_text = ["BUY", "SELL", "HOLD"][action]
    action_color = ["green", "blue", "red"][action]


     # Step 7: Render page with data
    return render_template("homepage.html",
        B=b_val,
        S=s_val,
        theta=theta,
        action=action_text,
        action_color=action_color,
        granularity=granularity, # Seconds: S5, S10, S15, S30 convert using js,
        instrument = instrument,
        signalMatrix=signal_matrix.to_dict(orient="records"),
        safe_timestamp = safe_timestamp)


@Arthur.route("/visualize/<instrument>", methods=["GET"])
def visualize_market(instrument):
    granularity = request.args.get("granularity" , default = "S15") #No Default. Only put in.

    # Step 1: Extract raw candle data
    candles, instrument, granularity = fetch_candles(instrument=instrument , granularity=granularity)

    # Step 2: Convert and clean
    tabular_data = convert_to_tabular(candles)
    cleaned_data, safe_timestamp = clean_tabular(tabular_data)

    # Step 3: Load Cleaned Data
    data_folder = get_today_folder(os.path.join(BASE_DIR, "data"))
    filename = load_cleaned_data(data_folder, cleaned_data, instrument, safe_timestamp)


    # Step 4: Generate Signal Matrix
    proto_signal = acquire_proto_signal(filename)
    signal_matrix = transform_to_signal(proto_signal)

    # Step 5: Load Signal Matrix
    signal_filename = load_signal_matrix(data_folder , signal_matrix, instrument, safe_timestamp)
    

    signal_dict = signal_matrix.to_dict(orient="records")


    return render_template("market_view.html",
                           signalMatrix=signal_dict,
                           instrument=instrument,
                           granularity=granularity)

if __name__ == "__main__":
    Arthur.run(debug=True)