### operators.py is a modular and agnostic framework that creates proprietary OOP
### in order to orchestrate hedge fund management systems. It interlocks
### all parts of the process with either hardcoded, dynamic (inner or UI pulled),
### or deferred parameters (args and/or kwargs).


## External Dependencies
from __future__ import annotations
import requests
from datetime import datetime , timedelta
from datetime import time as dt_time
from zoneinfo import ZoneInfo 
import os
import re
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event
import time
from enum import Enum

## Internal Dependencies
from utils.generic import fetch_candles
from utils.generic import convert_to_tabular , clean_tabular
from utils.generic import load_cleaned_data
from utils.generic import decisionrule , transform_to_signal
from utils.generic import load_signal_matrix

from utils.generic import extract_signal_matrix , extract_final_matrix
from utils.generic import finalize_records
from utils.generic import load_final_data

from utils.granularity import horizon_seconds, intake_seconds, check_granularity
from utils.http_helpers import OandaSession , json_or_raise


FILL = 0.00000000314      # transform_signal's warm-up fill value
ERROR_LIMIT = 3           # consecutive bad outcomes before holding
PRESSURE_Z = 4.0          # one-bar Margin change, in std devs of recent changes
PRESSURE_MIN_BARS = 20    # bars of change needed before judging
HOLD_HORIZONS = 3         # how long a hold lasts, in horizons


class OperationalState(Enum):
    OPEN = "OPEN"
    HOLDING = "HOLDING"
    CLOSED = "CLOSED"

class ArthurConfig():
    """
    ArthurConfig is a class that holds configuration information on the class Arthur being ran.
    
    Functions:
    __init__ : initialization function
    """

    def __init__(self , base_dir : str , mode : str , vendor : DataVendor , platform : Platform ,registry: dict,  instrument : str,
                 intake : str , horizon : str , zone : str , threadpool : ThreadPoolExecutor , state : OperationalState) -> None:
        """
        __init__ initializes ArthurConfig.
        It returns nothing.

        Inputs:
        self : self
        base_dir : str - base directory of storage
        mode : str - Mode of Instance
        vendor : DataVendor - Vendor being used for information
        platform : Platform - Platform being used to trade
        registry : dict - Vendor and Platform keys, stored on .env outside repository
        instrument : str -  Instrument being traded
        intake : str - Frequency of raw data stream
        horizon : str - Frequency of decision sample
        zone : str - Frequency of horizon samples used to make decision space UxV
        threadpool : ThreadPoolExecutor - Threading Pool for concurrent trading
        state : OperationalState - enumerated value for open holding or closed
        """

        self.base_dir = base_dir 
        self.mode = mode
        self.vendor = vendor
        self.platform = platform
        self.registry = registry
        self.instrument = instrument
        self.intake = intake
        self.horizon = horizon
        self.zone = zone
        self.threadpool = threadpool 
        self.operationalstate = state
        self.holdingbutton = None

        ## Requests Session for only one SSCS communication, lowering latency.
        self.session = OandaSession()      # was requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {registry['platform']}",
            "Accept-Datetime-Format": "RFC3339"
        })

class Arthur():
    """
    Arthur is a class that orchestrates the instance simulation.

    Functions:
    __init__ : initialization function
    _ignition_A: ignition function
    trader_cycle : one OODA loop for a trader
    market_cycle : one OODA loop for Arthur
    _acquire_market : ETL-Base
    _accounting : accounting dashboard summarizer function
    shutdown : shuts down the ThreadPoolExecutor
    """
    
    def __init__(self , cfg : ArthurConfig,  teamstructure : dict) -> None:
        """
        __init__ initializes Arthur.
        It returns nothing.

        Inputs:
        self : self
        cfg : ArthurConfig - the configuration of Arthur
        teamstructure : dict - the structure of managers and teams, currently uniform
        """
        
        self.config = cfg
        self.latest_signal = None

        num_managers = teamstructure["managers"] # Num managers, say 2
        num_traders = teamstructure["traders"] # Num traders, say 8
        
        self.managers = [RiskManager(f"RM{i}" ,self ,  self.config) for i in range(num_managers)]
        
        k = num_traders // num_managers

        self.traders = []

        for i, manager in enumerate(self.managers):
            
            team = []
            
            for j in range(k):
                trader_idx = i * k + j
                trader = Trader(
                    f"T{trader_idx}",
                    manager , 
                    manager.config)
                trader.set_trader_account("101-001-37152677-001") 
                trader.config.platform.set_accountID("101-001-37152677-001")
                # HARDCODED ACCOUNTID FOR NOW, ONLY 1
                # TRADER BECAUSE ONLY ONE ACCOUNT ID RIGHT NOW. THIS IS A BRYAN QUESTION.

                team.append(trader)
                
                self.traders.append(trader)
           
            manager.set_trader_team(team)

    def shutdown(self, wait: bool = True):
        """
        shutdown takes the ThreadPoolExecutor, lets it finish any work,
        and then shuts it down.

        Inputs:
        self : self
        wait : bool = True - Whether or not to wait on work
        """
        if self.config.threadpool:
            self.config.threadpool.shutdown(wait=wait)
            self.config.threadpool = None

    def _ignition_A(self):
        """
        _ignition_A is the total ignition on the Arthur Instance. It combines the managers'
        ignitions into one total "Open or Closed" check.
        It returns a string.

        Inputs:
        self : self
        """
    
        states = [m.ignition() for m in self.managers]

        if self.config.holdingbutton is not None:
            print("Holding Switch In Use: \n")
            print(f"Manual State is {self.config.holdingbutton}")
            return self.config.holdingbutton

        if any(s == OperationalState.OPEN for s in states):
            return OperationalState.OPEN
        if all(s == OperationalState.CLOSED for s in states):
            return OperationalState.CLOSED
        return OperationalState.HOLDING
    
    def trader_cycle(self , marketfile : str , manager : RiskManager , trader : Trader , kwargs : dict):
        """
        trader_cycle is an OODA loop with logging that is done per trader at horizon and is the basis
        of the trading engine for Arthur.
        It returns a JSON holding the decision, namely the action and status key value pairs.

        Input:
        self : self 
        marketfile : str - the file holding the signal
        manager : RiskManager - the manager for the trader
        trader : Trader - the trader in question
        kwargs : dict - keyword arguments for the context of the OODA loop
        """

        market = extract_signal_matrix(marketfile)["signal"]  # signal DataFrame

        decision = trader.decision_loop(manager, market) # {"action" : action , "status" : status}

        ## Every horizon: observe , measure , decide , act , then record. (OODA with logging)
        logs = extract_final_matrix(self.config.base_dir, trader) # A Trader records his results every cycle
        if logs["exists"]:
            final_matrix = finalize_records(market, trader,manager ,  record=logs["final"], **kwargs)["finalrecord"]
        else:
            final_matrix = finalize_records(market, trader, manager ,  **kwargs)["finalrecord"]

        load_final_data(final_matrix, logs["file"]) # CRUD for final records
        
        return decision # log JSON of action and status of action
    
    def _acquire_market(self):
        """
        _acquire_market performs ETL-Base.
        It returns a JSON.

        Input:
        self : self 
        """

        ## ETL-Base

        ## Get the Horizon as minutes
        check_granularity(self.config.intake, self.config.horizon)
        bars_per_horizon = horizon_seconds(self.config.horizon) // intake_seconds(self.config.intake)
        counter = 8 * self.config.zone * bars_per_horizon
        
        ## E1
        ## {"candles" : candles , "instrument" : instrument, "intake" : intake}
        candlebody = fetch_candles(self.config.mode , 
            self.config.vendor, self.config.platform , instrument = self.config.instrument, intake = self.config.intake,
            price = 'M' , count = int(counter) , timezone = 'UTC')

        ## T1
        ## Tabular list and then {"cleaned" : cleaned , "safe_ts" : safe_ts}
        tabular_data = convert_to_tabular(candlebody)
        cleaned_tabular = clean_tabular(tabular_data , self.config.horizon , self.config.intake)

        ## L1
        ## filename for the cleaned data
        if cleaned_tabular["cleaned"].empty :
            self.latest_signal = None
            return {"dict" : [] , "location" : ""}

        filename = load_cleaned_data(self.config.base_dir, cleaned_tabular["cleaned"], 
                    candlebody["instrument"], cleaned_tabular["safe_ts"])["file"]

        ## T2
        ## signal DataFrame
        signal_matrix = transform_to_signal(filename , self.config.zone)["signal"]

        ## L2
        ## filename for the signal data
        signal_filename = load_signal_matrix(self.config.base_dir , signal_matrix, 
                candlebody["instrument"], cleaned_tabular["safe_ts"])["file"]
        
        ## JS processing
        signal_dict = signal_matrix.to_dict(orient="records")
        self.latest_signal = signal_matrix
        return {"dict" : signal_dict , "location" : signal_filename} 
    

    def _market_cycle(self , marketfile : str):
        """
        _market_cycle is the singular process by which Arthur Instances run. Every horizon,
        the OODA loop for the entire instance is ran. Threading / Multiprocessing is used
        for concurrency and hardware optimization.
        It returns a JSON.

        Inputs:
        self : self 
        marketfile : str - the file location of the signal
        """
        if not marketfile:
            print("No market data this cycle, skipping trades.")
            return

        ## kwargs for context
        kwargs = {
        "t_o_e" : "order", "t_o_s" : 0,
        "t_t_e" : "trade", "t_t_s" : 0,
        "pricing" : "pricing", "pricing_sub" : 1,
        "acc_e" : "account", "acc_s" : 1,
        "transaction_e" : "transaction", "transaction_s" : 0}
    

        ## Flatten the managers and teams into trader manager pairs
        trader_pairs = [(m, t) for m in self.managers for t in m.team]    

        market_cycles = []

        start = time.time()
        
        ## Threading set up outside of cycle to only setup once, for concurrency
        futures = {
        self.config.threadpool.submit(self.trader_cycle, marketfile, m, t, kwargs) : m
        for (m, t) in trader_pairs
        }

        cycletime = time.time() - start

        for f in as_completed(futures): # As completed, append cycles
            decision = f.result()
            futures[f].record_outcome(decision)     # tell the manager how the trade went
            market_cycles.append({"decision": decision , "cycletime" : cycletime})



        cycles = pd.DataFrame(market_cycles)
    
        today = datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%d")
        present = datetime.now(ZoneInfo("UTC")).strftime("%H-%M-%S")
        end_dir = f"logs/{today}/logs_{present}.csv"
        full_path = os.path.join(self.config.base_dir,end_dir)

        ## Make the directory if it doesn't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        cycles.to_csv(full_path, index=False)



    def _accounting(self):
        """
        _accounting performs routine aggregate accounting for minimal information needed for guardian
        of Arthur.
        It returns a JSON.

        Inputs:
        self : self
        """

        ## Start with each trader, aggregate by account per trader
        traders = [t for m in self.managers for t in m.team]

        summaries = []
        account_summary = {}

        
        for trader in traders: # HTTP request construction
            summary_url = trader.config.platform.make_url(trader.config.mode , "account" , 2)

            response = self.config.session.get(url = summary_url)

            if response.status_code == 200:
                summaries.append(response.json()["account"])
            else:
                raise ValueError(f"Error {response.status_code}: {response.text}")
            
        ## Comprehension style ETL for account summary aggregation
        account_summary["NAV"] = sum([float(summary["NAV"]) for summary in summaries])
        account_summary["balance"] = sum([float(summary["balance"]) for summary in summaries])
        account_summary["openPositionCount"] = sum([float(summary["openPositionCount"]) for summary in summaries])
        account_summary["openTradeCount"] = sum([float(summary["openTradeCount"]) for summary in summaries])
        account_summary["pendingOrderCount"] = sum([float(summary["pendingOrderCount"]) for summary in summaries])
        account_summary["pl"] = sum([float(summary["pl"]) for summary in summaries])
        account_summary["RMs"] = len(self.managers)
        account_summary["minTraders"] = min([len(m.team) for m in self.managers])
        account_summary["maxTraders"] = max([len(m.team) for m in self.managers])
        account_summary["numTraders"] = len(traders)

        return {"summary" : account_summary} # Account summary JSON


class DataVendor:
    """
    DataVendor is a class that holds market information.

    Functions:
    __init__ : initialization function
    """
    
    def __init__(self):
        """
        __init__ initializes a DataVendor that is blank. There are base routes and endpoints on each API,
        as well as an authorization key.

        Inputs:
        self : self
        """ 
        self.authorization = None
        self.base = None
        self.endpoints = None



class Trader:
    """
    Trader is a class that trades and is controlled by the RiskManager class.

    Functions:
    __init__ - initialization function
    set_trader_account - setter for accountID
    close_trade - closes a trade
    cancel_order - cancels an order
    make_order - makes an order, currently oriented for long only
    decision_loop - decides and trades at horizon using signal
    """
    
    def __init__(self , name : str , manager : RiskManager , cfg : ArthurConfig):
        """
        __init__ initializes the Trader that makes the trades. A Trader takes parameters and is controlled
        by the RiskManager.

        Inputs:
        self : self
        manager : RiskManager - the manager of the trader
        cfg : ArthurConfig - the configuration of the Arthur Instance
        """
        
        self.name = name # Label for trader
        self.manager = manager # Gatekeeper for risk management
        self.config = cfg # Parameters inherited from Arthur
        self.accountID = None # accountID for the platform

    def set_trader_account(self , accountID : str):
        """
        set_trader_account sets the accountID for the trader. MVP uses 1 manager and minimal traders,
        which can be kept at one, so only volume is applied. This means that MVP
        can be hardcoded for a single accountID.
        It returns nothing.

        Inputs:
        self : self 
        accountID : str - the accountID to be used by the trader
        """

        ## This needs to be hardcoded in Arthur.py
        self.accountID = accountID
        
    def close_trade(self ,  trade_id):
        """
        close_trade closes a trade on the platform according to the trade_id , currently
        built for OANDA.
        It returns a JSON.

        Inputs:
        self : self 
        trade_id : str - the trading id for the trade in question.
        """
        
        ## Function partially inherited by the trading platform
        closure = self.config.platform.close_trade(self.config.mode , trade_id)

        fill = closure.get("orderFillTransaction")
        if fill:
            closed = fill.get("tradesClosed") or []
            return {"status" : "filled" , "raw" : closure , "order_id" : fill.get("orderID") ,
            "trade_id" : closed[0].get("tradeID") if closed else None}

        cancel = closure.get("orderCancelTransaction")
        if cancel:
            return {"status" : "left_open" , "raw" : closure , "order_id" : cancel.get("orderID")}

        return {"status" : "unknown" , "raw" : closure}

    def cancel_order(self , order_id):
        """
        This function is held here as a chassis for post MVP development when
        partial orders are utilized. 

        It is left incomplete.
        """

        ## Cancel order function inherited partially by platform.
        return self.config.platform.cancel_order(self.config.mode , order_id)


    def make_order(self , side : int , lot_size : float , reduce_only : bool):
        """
        make_order makes an order on long only conditions, and is followed by a close_trade action.
        FOK is used for rudimentary financial strategy.
        It returns a JSON.

        Input:
        self : self
        side : int - the buy or sell side
        lot_size : float - the size of the lot in notional currency
        reduce_only : bool - whether or not to reduce. side only gives long or short, reduce allows to 'sell long'
        """
        
        ## Make order function partially inherited by platform.
        order = self.config.platform.make_order(self.config.mode , self.config.instrument , side , lot_size , reduce_only)

        fill = order.get("orderFillTransaction")
        if fill:
            opened = fill.get("tradeOpened") or {}
            return {"status" : "filled" , "raw" : order , "order_id" : fill.get("orderID") ,
                    "trade_id" : opened.get("tradeID")}

        cancel = order.get("orderCancelTransaction")
        if cancel:
            return {"status" : "killed" , "raw" : order , "order_id" : cancel.get("orderID")}

        return {"status" : "unknown" , "raw" : order}
                
    def decision_loop(self , manager : RiskManager , market : pd.DataFrame):
        """
        decision_loop makes a single trade according to the decision call inside the signal matrix.
        It returns a JSON.

        Inputs:
        self : self
        manager : RiskManager - the manager of the trader
        market : pd.DataFrame - the signal of the market at current horizon
        """
        
        latest = market.iloc[-1]
        
        decisioncall = latest["ActionHat"] # 0 B 1 S 2 H

        action = ["BUY","SELL","HOLD"][int(decisioncall)]

        ## MVP is make orders and trade them. Sell immediately. Crude but will work.

        if action == "HOLD":
            ## Hold for now.
            return {"action" : action , "status" : "no operational change"}
            
        elif action == "SELL":
            ## Sell large majority on dump, accounts for asymmetric buying signals
            ## and works on stationary reverted to mean liquid FX pair like USDJPY.

            
            try:
                currentLong = self._long_units()
            except (ValueError , requests.RequestException) as e:
                return {"action" : action , "status" : {"status" : "error" , "detail" : str(e)}}

            if currentLong > 1000: # Crank minimum, hold onto very small min position
                num_traders = max(1 , len([t for m in manager.arthur.managers for t in m.team]))
                lot_size = (0.925 * currentLong) / num_traders
                # Traders get equal volume, threading means arithmetic not geometric

                return self._execute(action , -1 , lot_size , True)

            else:
                return {"action" : action , "status" : "no stock, wait for BUY"}
            
        else: # action == "BUY"
            
            ## Make a single order.
            static_lotsize = 1005

            (minsize , maxsize , bestsize) = manager.gate_lotsize()

            ## Static params, dynamic risk management.
            if manager.insiderange(minsize , maxsize, static_lotsize):
                lot_size = static_lotsize
            else: 
                lot_size = bestsize

            
            return self._execute(action , 1 , lot_size , False)

    def _long_units(self) -> float:
        """
        _long_units gives the units this trader is long in its instrument right now,
        0.0 if the account holds no position in it. OANDA lists an instrument under
        positions only once it has been traded, so a missing name means no position.
        It returns a float.

        Inputs:
        self : self
        """

        url = self.config.platform.make_url(self.config.mode , "account" , 1)
        body = json_or_raise(self.config.session.get(url = url) , "account details")

        for position in body.get("account" , {}).get("positions" , []):
            if position.get("instrument") == self.config.instrument:
                return float((position.get("long") or {}).get("units") or 0)
        return 0.0

    def _execute(self , action : str , side : int , lot_size : float , reduce_only : bool):
        """
        _execute places one order, closes the trade if one was opened, and turns platform errors
        into a logged status so one bad response doesn't crash the cycle. An 'error' status does
        not mean the order failed: after a timeout it may have filled, so check the account.
        """
        ## OANDA takes whole units for FX. Round down so a sell never exceeds what was meant.
        units = int(lot_size)
        if units < 1:
            return {"action" : action , "status" : {"status" : "skipped" , "detail" : f"lot size {lot_size} is under 1 unit"}}

        try:
            order = self.make_order(side = side , lot_size = units , reduce_only = reduce_only)
            if order["status"] == "filled" and order.get("trade_id"):
                return {"action" : action , "status" : self.close_trade(order["trade_id"])}
            return {"action" : action , "status" : order}
        except (ValueError , requests.RequestException) as e:
            return {"action" : action , "status" : {"status" : "error" , "detail" : str(e)}}

class RiskManager:
   
    devtime = True # devtime used to keep the trading bot open
    
    def __init__(self , name : str , arthur : Arthur , cfg : ArthurConfig):
        """
        __init__ initializes a RiskManager. RiskManagers are controlled by Arthur.
        It returns nothing.

        Inputs:
        self : self
        name : str - name of RiskManager
        arthur : Arthur - Arthur Instance
        cfg : ArthurConfig - Arthur Configuration
        """
        
        self.name = name # Name for logging
        self.arthur = arthur # RMs report to Arthur
        self.config = cfg # Config given from Arthur
        self.team = None # Team iterated during Arthur construction
        self.bad_streak = 0
        self.hold_until = None
        self.anomaly_reason = None

    def record_outcome(self , decision : dict):
        """
        record_outcome counts consecutive bad trading outcomes for anomaly detection.
        Inputs:
        decision : dict - {"action" , "status"} returned by Trader.decision_loop
        """
        status = decision.get("status")
        kind = status.get("status") if isinstance(status , dict) else None
        self.bad_streak = self.bad_streak + 1 if kind in ("error" , "unknown" , "left_open") else 0

    def _pressure_shift(self):
        """
        _pressure_shift returns a reason string if the newest one-bar change in Margin is extreme
        compared with recent changes, else None.
        """
        sig = getattr(self.arthur , "latest_signal" , None)
        if sig is None or "Margin" not in sig:
            return None
        margin = sig["Margin"]
        margin = margin[(margin - FILL).abs() > 1e-12]       # drop warm-up fill rows
        dm = margin.diff().dropna()
        if len(dm) < PRESSURE_MIN_BARS:
            return None
        sd = dm.iloc[:-1].std()
        if not sd or pd.isna(sd):
            return None
        z = abs(dm.iloc[-1]) / sd
        return f"Margin moved {z:.1f} sd in one bar" if z > PRESSURE_Z else None

    def detect_minor_anomaly(self) -> bool:
        """
        detect_minor_anomaly gives True while the manager should hold: repeated error or unknown
        outcomes, or a sudden shift in buying / selling pressure. A hold lasts HOLD_HORIZONS horizons.
        It returns a bool.
        """
        now = datetime.now(ZoneInfo("UTC"))

        if self.hold_until is not None:
            if now < self.hold_until:
                return True
            self.hold_until = None          # hold over, start fresh
            self.bad_streak = 0
            self.anomaly_reason = None

        reason = (f"{self.bad_streak} consecutive bad outcomes"
                if self.bad_streak >= ERROR_LIMIT else self._pressure_shift())

        if reason:
            self.anomaly_reason = reason
            self.hold_until = now + timedelta(seconds = HOLD_HORIZONS * horizon_seconds(self.config.horizon))
            print(f"[ANOMALY] {self.name} holding until {self.hold_until:%H:%M:%S} UTC: {reason}")
            return True
        return False
    
    def in_budget(self): 
        """
        in_budget calculates whether or not the team under management is under budget.
        It returns a bool.

        Inputs:
        self : self      
        """
              
        max_trades = 15
        static_lotsize = 1000
                
        ## Query closed trades today (for whole account/team)
        running_trades = 0
        for trader in self.team:
            running_trades += trader.config.platform.count_trades(trader.config.mode , trader.config.instrument)

        spent_today = running_trades*static_lotsize
        
        ## Budget scales with team size
        budget = max_trades * len(self.team)*static_lotsize
        
        return spent_today < budget # static lotsize reduces to trades today < maxnumtrades

    def fx_market_open(self, now_utc: datetime) -> bool:
        """
        fx_market_open gives True iff the spot FX week is open: from Sunday 17:00 to
        Friday 17:00 America/New_York (the OANDA week). Using New York time keeps the
        UTC open/close moving with daylight saving time (22:00 UTC in winter, 21:00 in summer).
        It returns a bool.

        Inputs:
        now_utc : datetime - timezone-aware current time
        """

        ny = now_utc.astimezone(ZoneInfo("America/New_York"))
        weekday, t = ny.weekday(), ny.time()
        rollover = dt_time(17, 0)

        if weekday == 4:           # Friday: open until 17:00 NY
            return t < rollover
        if weekday == 5:           # Saturday: closed
            return False
        if weekday == 6:           # Sunday: open from 17:00 NY
            return t >= rollover
        return True                # Monday to Thursday: open


    def in_bhe_hours(self) -> bool:
        """
        in_bhe_hours gives True iff USD/JPY can be traded now. The FX market must be open,
        and unless devtime is on, it must also be Mon-Fri 06:55-15:00 America/Chicago.
        It returns a bool.

        Inputs:
        self : self
        """

        now_utc = datetime.now(ZoneInfo("UTC"))

        
        
        if not self.fx_market_open(now_utc):
            return False

        if self.devtime:
            return True
        

        chicago = now_utc.astimezone(ZoneInfo("America/Chicago"))
        return chicago.weekday() < 5 and dt_time(6, 55) <= chicago.time() < dt_time(15, 0)
    
    

    def ignition(self):
        """
        ignition gives the status on whether or not BHE is in operational hours and in budget.
        It returns a string.

        Inputs:
        self: self
        """
        
        if self.in_bhe_hours() and self.in_budget():
            if self.detect_minor_anomaly():
                return OperationalState.HOLDING
            else:
                return OperationalState.OPEN
        else:
            return OperationalState.CLOSED

        
        
    def set_trader_team(self , team : list):
        """
        set_trader_team sets the team for the RiskManager.
        It returns nothing.

        Inputs:
        self : self 
        team : list - the list of traders for the manager
        """
        
        self.team = team

    def insiderange(self , start : float , end : float , val : float) -> bool:
        """
        insiderange takes a real number interval and gives the truth value of if the value
        in question is inside that range inclusively.
        It returns a bool.

        Inputs:
        self : self
        start : float - start value
        end : float - end value
        val : float - value in question
        """
        
        return (start <= val <= end)

    
    def gate_lotsize(self):
        """
        gate_lotsize gates the lotsize of the buy side decisions statically.
        It returns a tuple.

        Inputs:
        self : self
        """
        
        ## Currently very static but becomes dynamic. 
        lotsizemin = 1000 
        lotsizemax = 2000
        lotsizebest = 1000 
        

        return (lotsizemin , lotsizemax , lotsizebest)



class Platform:
    """
    Platform is a class that interfaces with trading platforms. It is left blank.

    Functions:
    __init__
    """
    
    def __init__(self):
        """
        __init__ initializes the class Platform.
        It returns nothing.

        Inputs:
        self : self
        """
        
        self.authorization = None
        self.base = None
        self.endpoints = None

    def make_url(self , mode : str, endpoint: str , subEndpoint: int, **kw):
        """
        make_url takes inputs on a Platform and makes a url for HTTP requests.
        It returns a string.

        Inputs:
        self : self
        mode : str - the mode of the Instance
        endpoint : str - the endpoint on the API
        subEndpoint : int - the number in the list of the subendpoints for proprietary OOP Interface
        kw : dict - key value pairs for optional arguments
        """

        context = {}

        context.update(**kw)
 
        return self.base[mode] + self.endpoints[endpoint][subEndpoint].format(**context)


class OandaTrading(Platform): 
    """
    OandaTrading(Platform) is a class that inherits from Platform. It is the class that interfaces with 
    Oanda's REST API. For MVP it is both the vendor and the platform. For modularity and other reasons, Oanda
    is to be siloed and not used. It would be simpler and easier to understand if two separate interfaces
    were used. Then a vendor could be called, and a platform could be called. Cost will be negligible.

    Functions:
    __init__ : initialization function
    set_accountID : setter for accountID
    set_instrument : setter for instrument
    set_orderSpecifier : setter for orderSpecifier
    set_tradeSpecifier : setter for tradeSpecifier
    make_url : url HTTP constructor
    close_trade : function that closes trades
    cancel_order : function that cancels orders
    make_order : function that makes orders
    count_trades : function that counts todays closed buy trades for RiskManager budget function
    """
    
    def __init__(self, api_key : str):
        self.authorization = api_key # Authorization
        self.accountID = None # Point to Nothing , Set to Anything
        self.instrument = None # Point to Nothing, Set to Anything
        self.config = None
        self.orderSpecifier = None # No particular order per account , Set to Anything
        self.tradeSpecifier = None # No particular order per account , Set to Anything

        demo_base = "https://api-fxpractice.oanda.com"
        live_base = "https://api-fxtrade.oanda.com"

        Account = ["/v3/accounts",
                   "/v3/accounts/{accountID}",
                   "/v3/accounts/{accountID}/summary",
                   "/v3/accounts/{accountID}/instruments",
                   "/v3/accounts/{accountID}/configuration",
                   "/v3/accounts/{accountID}/changes"]
        
        Instrument = ["/v3/instruments/{instrument}/candles"]
        
        Order = ["/v3/accounts/{accountID}/orders",
                 "/v3/accounts/{accountID}/pendingOrders",
                 "/v3/accounts/{accountID}/orders/{orderSpecifier}",
                 "/v3/accounts/{accountID}/orders/{orderSpecifier}/cancel",
                 "/v3/accounts/{accountID}/orders/{orderSpecifier}/clientExtensions"]
        
        Trade = ["/v3/accounts/{accountID}/trades",
                 "/v3/accounts/{accountID}/openTrades",
                 "/v3/accounts/{accountID}/trades/{tradeSpecifier}",
                 "/v3/accounts/{accountID}/trades/{tradeSpecifier}/close",
                 "/v3/accounts/{accountID}/trades/{tradeSpecifier}/clientExtensions",
                 "/v3/accounts/{accountID}/trades/{tradeSpecifier}/orders"]
        
        Position = ["/v3/accounts/{accountID}/positions",
                    "/v3/accounts/{accountID}/openPositions",
                    "/v3/accounts/{accountID}/positions/{instrument}",
                    "/v3/accounts/{accountID}/positions/{instrument}/close"
                    ]
        
        Transaction = ["/v3/accounts/{accountID}/transactions"]
        
        Pricing = ["/v3/accounts/{accountID}/candles/latest",
                   "/v3/accounts/{accountID}/pricing",
                   "/v3/accounts/{accountID}/pricing/stream",
                   "/v3/accounts/{accountID}/instruments/{instrument}/candles"]
        
        self.base = {
            "Demo" : demo_base,
            "Live" : live_base
        }

        self.endpoints = {
            "account" : Account,
            "instrument" : Instrument,
            "order" : Order,
            "trade" : Trade,
            "position" : Position,
            "transaction": Transaction,
            "pricing": Pricing
        }

    def set_accountID(self , accountID : str):
        """
        set_accountID is a setter function for the accountID parameter.
        It returns nothing.

        Inputs:
        self : self
        """ 
        
        self.accountID = accountID
    
    def set_instrument(self, instrument : str): 
        """
        set_instrument is a setter function for the instrument parameter.
        It returns nothing.

        Inputs:
        self : self
        """

        self.instrument = instrument
    
    def set_orderSpecifier(self , orderSpecifier: str):
        """
        set_orderSpecifier is a setter function for the orderSpecifier parameter.
        It returns nothing.

        Inputs:
        self : self
        """

        self.orderSpecifier = orderSpecifier
    
    def set_tradeSpecifier(self , tradeSpecifier: str): 
        """
        set_tradeSpecifier is a setter function for the tradeSpecifier parameter.
        It returns nothing.

        Inputs:
        self : self
        """

        self.tradeSpecifier = tradeSpecifier

    def set_config(self , cfg : ArthurConfig):
        """
        set_config is a setter function for the configuration of the instance
        in the OandaTrading object.

        Inputs:
        self : self
        cfg : ArthurConfig - the configuration of the instance
        """
    
        self.config = cfg

    def make_url(self , mode : str, endpoint: str , subEndpoint: int , **kw):
        """
        make_url for OandaTrading is the inherited version of make_url.
        It returns a string.

        Inputs:
        self : self
        mode : str - the mode of the Instance
        endpoint : str - the endpoint on the API
        subEndpoint : int - the number in the list of the subendpoints for proprietary OOP Interface
        kw : dict - key value pairs for optional arguments
        """


        context = {
            "accountID" : self.accountID,
            "instrument": self.instrument,
            "orderSpecifier" : self.orderSpecifier,
            "tradeSpecifier" : self.tradeSpecifier
        }

        context.update(kw) # kw arguments update context

        return self.base[mode] + self.endpoints[endpoint][subEndpoint].format(**context)
    
    def close_trade(self , mode : str , trade_id : str):
        """
        close_trade closes a trade on OANDA.
        It returns a JSON.

        Inputs:
        self : self 
        mode : str - mode of Arthur being ran
        trade_id : str - trade id for OANDA REST API
        """

        url = self.make_url(mode , "trade" , 3, tradeSpecifier = trade_id)

        body = {
                "units" : "ALL" # Close a trade immediately, small lots -> easy trade closures
        }
    
        response = self.config.session.put(url, json = body)
        
        if response.status_code == 200:
            return response.json() 
        else:
            raise ValueError(f"Error {response.status_code}: {response.text}")

    def cancel_order(self , mode : str , order_id : str):
        """
        cancel_order cancels an order on the OANDA REST API. It is blank because MVP does not cancel orders.
        It is left incomplete.
        """
        
        return None

    def make_order(self, mode : str, instrument : str , side : int, lot_size : int , reduce_only : bool = False):
        """
        make_order makes an order on the OANDA REST API.
        It returns a JSON.

        Inputs:
        self : self
        mode : str - mode for Arthur instance running
        instrument : str - the instrument being traded
        side : int - the trade side
        lot_size : int - the amount of the lot in notional currency
        reduce_only : bool - whether or not to reduce the units. Used for long only trades, where side sell is
        then True, and reduce the long, as opposed to the sell only then starts a short
        """
        
        url = self.make_url(mode,"order",0)
        
        body = {
                    "order": {
                        "units": str(side*lot_size),
                        "instrument": instrument,
                        "timeInForce": "FOK", # MVP uses FOK because it is simpler
                        "type": "MARKET",
                        "positionFill": "REDUCE_ONLY" if reduce_only else "DEFAULT"
                    }
                }

        response = self.config.session.post(url, json = body)
        
        if response.status_code == 201:
            return response.json() 
        else:
            raise ValueError(f"Error {response.status_code}: {response.text}")
        
    def count_trades(self , mode: str, instrument : str):
        """
        count_trades counts the trades that are closed buys for today via the OANDA REST API.
        It returns an integer.

        Inputs:
        self : self
        mode : str - the mode of Arthur being ran
        instrument : the instrument being traded  
        """
        
        url = self.make_url(mode , "trade", 0)

        params = {
            "state" : "CLOSED",
            "instrument" : instrument,
            "count" : 500 # Maximum allowed, filter manually by time, other platform might 
            # allow more.
        }

        request_body = self.config.session.get(url = url , params = params)

        if request_body.status_code == 200:
            last_count_trades = request_body.json()["trades"]
        else:
            raise ValueError(f"Error {request_body.status_code} {request_body.text}")
        
        utc = ZoneInfo("UTC")
        now_utc = datetime.now(utc)
        midnight_utc = datetime(year = now_utc.year , month = now_utc.month , day = now_utc.day , tzinfo = utc)

        def parse_rfc3339_to_utc(rfc3339: str) -> datetime:
            """Convert OANDA RFC3339 string to UTC datetime (drop nanos)."""
            base = rfc3339.rstrip("Z")
            if "." in base:
                base = base.split(".")[0]
            return datetime.fromisoformat(base).replace(tzinfo=utc)

        ## Local filter: today
        count = 0
        for t in last_count_trades:
            ct = t.get("closeTime")
            if not ct:
                continue
            ct_utc = parse_rfc3339_to_utc(ct)
            if midnight_utc <= ct_utc <= now_utc and float(t["currentUnits"]) > 0: # Only buy, because count the 
                # money spent via buys. Can be changed.
                count += 1

        return count