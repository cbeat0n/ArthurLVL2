### Arthur.py is a modular and agnostic framework that utilizes microwebframeworks
### and Python / Javascript for End to End turnkey hedge fund systems.
### The main differentiating factor at MVP is pre compliance, logging, speed to execution,
### accuracy of execution, and modularity / agnosticity so evolution
### is legal but outpaces regulation.

## External Dependencies
from dotenv import load_dotenv # Env hides API Key
import signal # Library for Server signals
from datetime import datetime , timezone # Library for time objects
from flask import Flask, render_template, request # Library for common MWF
from flask import jsonify , redirect, url_for # Library for common MWF
import pandas as pd # Library for data manipulation
import time # Library for time
import os , glob , re # Library for operating system communication and pattern matching
from threading import Event , Lock , RLock # Library for flagging black swans and guarding startup
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
from utils.instruments import SUPPORTED
from utils.scheduler import CycleScheduler

## Load environment variables from the .env file
load_dotenv()

## Initialize Flask App and globally hardcoded parameters.
ArthurFlask = Flask(__name__)
ArthurFlask.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0 # static files are re-checked every load, so edited CSS is never stale


@ArthurFlask.after_request
def guard_headers(response):
    """
    guard_headers adds safety headers to every response. Pages and data are never cached, so a
    reload or the back button can't show an old account or an old switch state. The pages can't
    be placed inside another site's frame, which stops a hidden overlay clicking the switch.
    It returns the response.

    Inputs:
    response : Response - the response Flask is about to send
    """

    if not request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response

GLOBAL_STATE = {"Arthur" : None , "Scheduler" : None} # Arthur, its scheduler and state start at None
START_LOCK = Lock() # one startup at a time
STATE_LOCK = RLock() # guards reads and writes of GLOBAL_STATE. Held for a moment only, never across a cycle or a network call.
STOP_WHEN_CLOSED = True # True: stop the server when CLOSED. False: stay up and wait for the next cycle
ALLOW_LIVE = False # True: the homepage form may start Live mode. Keep False until Live keys, account and trading are ready. Change it here only, never from a URL or form.
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data")
REGISTRY = {"vendor" : None , "platform" : os.getenv("OANDADEMOAPIKEY")} # Can keep .env separate
VENDOR = DataVendor() # Blank because MVP uses Oanda
ARTHUR_SERVER="dev"

@ArthurFlask.route("/", methods=["GET"])
def home():
    """
    home sets up a simple frontend intake page where the user is able to input parameters
    to modify Arthur. It is MVP friendly and scalable.
    It returns a rendered template.

    Inputs:
    None
    """
    
    return render_template("homepage.html", allowLive = ALLOW_LIVE)


def stop_server():
    """
    stop_server stops the scheduler, shuts down the thread pool, and then ends the process.
    It is a plain function and not a web route, so nothing outside the process can call it.
    It is safe to call from the scheduler's own thread. This is the one place that will also
    cancel orders and close trades once the emergency stop is built.
    It returns nothing.

    Inputs:
    None
    """

    with STATE_LOCK:
        scheduler = GLOBAL_STATE.get("Scheduler")
        arthur = GLOBAL_STATE.get("Arthur")

    if scheduler is not None:
        scheduler.stop() # does not wait on itself when called from its own thread

    if arthur is not None:
        arthur.shutdown() # Stops thread pooling

    print("Server shutting down...", flush = True)
    os.kill(os.getpid(), signal.SIGTERM) # on Windows this ends the process at once


def scheduled_cycle():
    """
    scheduled_cycle is what the scheduler runs once per horizon. It returns False to stop the
    scheduler, anything else to continue.

    Inputs:
    None
    """

    with STATE_LOCK:
        arthur = GLOBAL_STATE["Arthur"]

    state = arthur.run_once() # not under the lock: a cycle can take seconds

    with STATE_LOCK:
        GLOBAL_STATE["State"] = state

    if state == OperationalState.CLOSED and STOP_WHEN_CLOSED:
        print("Trading out of bounds for current operations.")
        stop_server()
        return False
    return True


@ArthurFlask.route("/start", methods = ["POST"])
def start_arthur():
    """
    start_arthur builds Arthur and starts its scheduler from the homepage form. It is POST only, so a
    refresh, prefetch or old link can't start the bot. Live mode is off unless ALLOW_LIVE is True in this
    file, and even then it needs the confirmation box ticked.
    Trading then runs on the scheduler's own thread, once per horizon, with no page open.
    It redirects to the dashboard, or returns a 400 for a bad form.

    Inputs:
    None
    """

    global GLOBAL_STATE , BASE_DIR , VENDOR , REGISTRY

    with START_LOCK: # two submits at once must not start two bots

        with STATE_LOCK:
            running = GLOBAL_STATE["Arthur"] is not None

        if running: # already running, never rebuild it
            return redirect(url_for("enter_vault") , code = 303)

        form = request.form
        instrument = form.get("currencyPair", "USD_JPY")
        intake = form.get("intake", "S5")
        horizon = form.get("horizon", "M1Q0")
        mode = form.get("mode", "Demo")

        try:
            zone = int(form.get("zone", "10"))
            managers = int(form.get("Managers", 1))
            traders = int(form.get("traders", 1))
        except ValueError:
            return "zone, Managers and traders must be whole numbers.", 400

        if zone < 1 or managers < 1 or traders < 1:
            return "zone, Managers and traders must be at least 1.", 400

        if mode not in ("Demo", "Live"):
            return f"Unknown mode {mode!r}.", 400

        if mode == "Live" and not ALLOW_LIVE:
            return "Live mode is switched off in Arthur.py (ALLOW_LIVE = False).", 403

        if mode == "Live" and form.get("confirmLive") != "yes":
            return "Live mode uses real money. Go back and tick the confirmation box.", 400

        ## Refuse a bad instrument or pairing now, not inside the scheduler thread every cycle
        if instrument not in SUPPORTED:
            return f"Unsupported instrument {instrument!r}. Supported: {', '.join(SUPPORTED)}", 400
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
        arthur = Arthur(cfg = config , teamstructure = {"managers" : managers , "traders" : traders})

        ## One cycle per horizon, on the horizon boundary, until the process ends
        scheduler = CycleScheduler(period_s = horizon_seconds(horizon), run_cycle = scheduled_cycle)

        ## Publish all three together, so no page ever sees Arthur without its scheduler
        with STATE_LOCK:
            GLOBAL_STATE["Arthur"] = arthur
            GLOBAL_STATE["State"] = config.operationalstate
            GLOBAL_STATE["Scheduler"] = scheduler

        scheduler.start()

    return redirect(url_for("enter_vault") , code = 303)


@ArthurFlask.route("/accounting", methods = ["GET","POST"])
def enter_vault():
    """
    enter_vault shows the dynamic accounting dashboard and handles the 4 way switch. It never
    starts or runs anything: Arthur is started by /start, and trades on the scheduler's thread.
    With no bot running it sends you back to the homepage.
    It returns a rendered template.

    Inputs:
    None
    """

    with STATE_LOCK:
        arthur = GLOBAL_STATE["Arthur"]
        scheduler = GLOBAL_STATE["Scheduler"]

    if arthur is None or scheduler is None: # nothing started yet: view only, so go to the form
        return redirect("/")

    ## 4 way switch, not manual in hand switch
    if request.method == "POST":
        value = request.form.get("autoSwitch")
        if value == "AUTOMATED":
            arthur.config.holdingbutton = None
        elif value in OperationalState.__members__:
            arthur.config.holdingbutton = OperationalState[value]
        ## Post, redirect, get: a reload of the page after this can never send the switch again
        return redirect(url_for("enter_vault") , code = 303)

    manual = arthur.config.holdingbutton.name if arthur.config.holdingbutton is not None else "AUTOMATED"

    ## Accounting Information, plus how the scheduler is doing
    account_summary = arthur._accounting()["summary"]
    ## What each manager is doing about anomalies, and whether the Chicago hours window is ignored
    manager_cards = {"Devtime (Chicago hours ignored)" : "on" if arthur.managers[0].devtime else "off"}
    for m in arthur.managers:
        manager_cards[f"{m.name} hold"] = (f"until {m.hold_until:%H:%M:%S} UTC: {m.anomaly_reason}"
                                           if m.hold_until else "none")

    account_summary = {**account_summary ,
                       "Bot state" : arthur.config.operationalstate.value ,
                       "Manual switch" : manual ,
                       **manager_cards ,
                       **scheduler.status()}

    return render_template("vault_storage.html", 
                           accountSummary = account_summary,
                           instrument = arthur.config.instrument,
                           refreshRate = arthur.config.horizon,
                           zone = arthur.config.zone,
                           manualSwitch = manual)

def latest_signal_records(instrument):
    """Newest signal file for the instrument as a list of records, or None."""
    if instrument not in SUPPORTED:
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
    horizon = request.args.get("horizon", "M1Q0")
    if not re.fullmatch(r"M\d+Q[0-3]", horizon): # the form MXQN, e.g. M1Q0
        return "Bad horizon. Use the form MXQN, e.g. M1Q0.", 400
    return render_template("market_view.html", signalMatrix=records, instrument=instrument,
                           refreshRate=horizon,
                           zone=request.args.get("zone"))

@ArthurFlask.route("/visualize/data", methods=["GET"])
def visualize_data():
    records = latest_signal_records(request.args.get("instrument", ""))
    if records is None:
        return jsonify({"error": "no signal data"}), 404
    return jsonify(records)


## Werkzeug for Development (slower, construction, setting up skeleton into chassis)
## Waitress for Production -> Move to Gunicorn and Docker for full multiple computers
## When basis is ready, Waitress is stateless and spits out every last incorrect assertion.

if __name__ == "__main__":
    print("Server =", ARTHUR_SERVER, flush=True)
    if ARTHUR_SERVER == "dev":
        ArthurFlask.run(debug=False)   # Werkzeug: banner, readable errors
    else:
        from waitress import serve
        print("Arthur is up at http://127.0.0.1:5000/  (Ctrl+C to stop)", flush=True)
        serve(ArthurFlask, host="127.0.0.1", port=5000, threads=6)
