### operators.py is a modular and agnostic structure outside of the proprietary
### ETL-Base and ETL-Auto that allows for construction and simulation of
### 4 main operators, the data vendor, the trader(s), the risk manager, and the 
### trading platform. Arthur simulates the interactions between these either in historical
### or is the communicative body of work that allows for trading to take place.
### Arthur decides the how and why: "Let's go bet on horses, here's the structure."

## Although operators.py is for modularity and agnosticity, there is also
## a cloud and cybersecurity element to it. Currently API Keys are static and
## stored proprietarily but operators or some other system will hold them 
## with more safety in mind.


# Dependencies
from __future__ import annotations
import requests
from datetime import date , datetime
from zoneinfo import ZoneInfo  # Python 3.9+
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event
import time

# Internal Dependencies
from utils.extract_quantitative import fetch_candles
from utils.transform_cleaned import convert_to_tabular , clean_tabular
from utils.load_cleaned import get_today_folder , load_cleaned_data
from utils.transform_signal import acquire_proto_signal , decisionrule , transform_to_signal
from utils.load_signal import load_signal_matrix

from utils.extract_auto import extract_signal_matrix , extract_final_matrix
from utils.transform_auto import finalize_records
from utils.load_auto import load_final_data


class ArthurConfig():
    def __init__(self , mode : str , vendor : DataVendor , platform : Platform , instrument : str,
                 intake : str , horizon : str):
        self.mode = mode
        self.vendor = vendor
        self.platform = platform
        self.registry = None
        self.instrument = instrument
        self.intake = intake
        self.horizon = horizon
        self.stop_event = Event()  # allows graceful shutdown of all threads

class Arthur():
    def __init__(self , cfg : ArthurConfig,  managers : list , base_dir : str):
        self.config = cfg
        self.managers = managers # Threading on trades MVP later Multiprocessing
        # Managers all share same instrument per instance of Arthur.
        # Managers might have different strategies. MVP is one Manager.
        self.base_dir = base_dir
    
    def set_registry(self , registry : dict):
        # registry is {vendorKey : key1 , platformKey : key2}
        self.config.registry = registry


    def _ignition_A(self):
        states = [m.ignition() for m in self.managers]
        return "OPEN" if all(s == "OPEN" for s in states) else "CLOSED"
    
    def trader_cycle(self , marketfile : str , manager : RiskManager , trader : Trader , kw : dict):
        """One trader’s cycle (one pass). Call in a loop if you want continuous intake."""
        # 1) Read latest signal matrix (shared file, but re-read per intake cycle is fine)
        market = extract_signal_matrix(marketfile)  # pd.DataFrame

        # 2) Trader decides (and may place orders)
        trader.decision_loop(manager, market)

        # 3) Finalize & persist telemetry
        logs = extract_final_matrix(self.base_dir, trader)
        if logs["exists"]:
            final_matrix = finalize_records(market, trader, record=logs["df"], **kw)
        else:
            final_matrix = finalize_records(market, trader, **kw)

        load_final_data(final_matrix, logs["file"])
        return trader.name
    
    def _acquire_market(self):
        ## ETL-Base : ETLTL
        # Make internal names same as ETL-Base
        # Step 1: Extract raw candle data
        candles, instrument, intake = fetch_candles(instrument=self.config.instrument, intake = self.config.intake)

        # Step 2: Convert and clean
        tabular_data = convert_to_tabular(candles)
        cleaned_data, safe_timestamp = clean_tabular(tabular_data , horizon = self.config.horizon)

        # Step 3: Load Cleaned Data
        data_folder = get_today_folder(os.path.join(self.base_dir, "data"))
        filename = load_cleaned_data(data_folder, cleaned_data, instrument, safe_timestamp)


        # Step 4: Generate Signal Matrix
        proto_signal = acquire_proto_signal(filename)
        signal_matrix = transform_to_signal(proto_signal)

        # Step 5: Load Signal Matrix
        signal_filename = load_signal_matrix(data_folder , signal_matrix, instrument, safe_timestamp)
        

        signal_dict = signal_matrix[["timestamp" , "value",
                        "B","S","theta","mu","ActionHat"]].to_dict(orient="records")

        return signal_dict , signal_filename # Is a list of dictionaries , JS needs JSON
    

    def _market_cycle(self , marketfile : str):
        # market cycle 
        # on intake granularity, run arthur into rms into each trader and have them make 
        # their decision of B S H.

        # kwargs must be instantiated. Context matters, these work on OANDA.
        kw = {
        "t_o_e" : "order",
        "t_o_s" : 0,
        "t_t_e" : "trade",
        "t_t_s" : 0,
        "pricing" : "pricing",
        "pricing_sub" : 1,
        "acc_e" : "account",
        "acc_s" : 1,
        "transaction_e" : "transaction",
        "transaction_s" : 0
    }
    

        # Flatten all traders across all managers
        trader_pairs = [(m, t) for m in self.managers for t in m.team]
        if not trader_pairs:
            return

        start = time.time()
        # Size pool to number of traders (cap if needed)
        with ThreadPoolExecutor(max_workers=len(trader_pairs)) as pool:
            futures = [
                pool.submit(self.trader_cycle,marketfile, m, t, kw)
                for (m, t) in trader_pairs
            ]
            for f in as_completed(futures):
                name = f.result()  # will raise if worker threw
                #print(f"[cycle] {name} done")
                # Timing takes up space. Contextual.

        #print(f"[cycle] duration: {time.time() - start:.3f}s")
        # Timing takes up space. Contextual.

    def _accounting(self):
        # go through account/summary and aggregate for account wide.
        # also give number of rm, traders per team , total traders that day.

        traders = [t for m in self.managers for t in m.team]

        summaries = []
        account_summary = [{}]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.registry["platformKey"]}",
            "Accept-Datetime-Format" : "RFC3339"
        }
        
        for trader in traders:
            summary_url = trader.platform.make_url(trader.mode , "account" , 2)

            response = requests.get(url = summary_url, headers = headers)

            if response.status_code == 200:
                summaries.append(response.json()["account"])
            else:
                raise ValueError(f"Error {response.status_code}: {response.text}")
            
        account_summary[0]["NAV"] = sum([float(summary["NAV"]) for summary in summaries])
        account_summary[0]["balance"] = sum([float(summary["balance"]) for summary in summaries])
        account_summary[0]["openPositionCount"] = sum([float(summary["openPositionCount"]) for summary in summaries])
        account_summary[0]["openTradeCount"] = sum([float(summary["openTradeCount"]) for summary in summaries])
        account_summary[0]["pendingOrderCount"] = sum([float(summary["pendingOrderCount"]) for summary in summaries])
        account_summary[0]["pl"] = sum([float(summary["pl"]) for summary in summaries])
        account_summary[0]["RMs"] = len(self.managers)
        account_summary[0]["minTraders"] = min([len(m.team) for m in self.managers])
        account_summary[0]["maxTraders"] = max([len(m.team) for m in self.managers])
        account_summary[0]["numTraders"] = len(traders)

        return account_summary


# Data Vendor (vendor, bank, other digital ClientServer streams)
class DataVendor: # Generic Data Vendor containing {x_j} None collection of all 
    def __init__(self): # possible base and endpoints. Specific Data Vendors use subcollection.
        self.base = None
        self.endpoints = None
# Trader(s) that decides what and when (timing) based on instrument and regime from vendor
# trade function make it static with dynamic lot size
class Trader:
    def __init__(self ,name : str, mode : str , instrument : str , platform : Platform , intake : str , horizon : str):
        self.name = name
        self.mode = mode
        self.instrument = instrument
        self.intakeGranularity = intake # RM puts this in
        self.horizonGranularity = horizon # RM puts this in
        self.platform = platform
        self.accountID = None # move accountID focus from platform to trader, 
        # one platform multiple traders and then the traders call the platform with their 
        # id.

    def set_trader_account(self , accountID : str):
        self.accountID = accountID
        
    def close_trade(self ,  trade_id):
        # close_trade function partially inherited based on platform

        closure = self.platform.close_trade(self.mode , trade_id)

        if closure.get("orderFillTransaction"):
            return {"status" : "filled" , "raw" : closure , 
                    "order_id" : closure["orderFillTransaction"]["orderID"] , 
                     "trade_id" : closure["orderFillTransaction"]["tradesClosed"][0]["tradeID"] }
        elif closure.get("orderCancelTransaction"):
            return {"status" : "left_open" , "raw" : closure , "order_id" : closure["orderCancelTransaction"]["orderID"]}
        

    def cancel_order(self , order_id):
        # cancel_order function partially inherited based on platform

        # FOK with market / lot size means no rolling orders to cancel... yet.
        # leverage and high liquidity could mean rolling orders to cancel
        # for now dynamic make order that allows for "this got cancelled"

        return self.platform.cancel_order(self.mode , order_id)


    def make_order(self , side , lot_size : float , reduce_only : bool):
        # make_order function partially inherited based on platform

        order = self.platform.make_order(self.mode , self.instrument , side , lot_size , reduce_only)

        if order.get("orderFillTransaction") :
            return {"status": "filled" , "raw" : order , 
                    "order_id" : order["orderFillTransaction"]["orderID"],
                    "trade_id" : order["orderFillTransaction"]["tradeOpened"]["tradeID"]}
        
        elif order.get("orderCancelTransaction"):
            return {"status" : "killed" , "raw" : order , "order_id" : order["orderCancelTransaction"]["orderID"]}
        
        
    def decision_loop(self , manager : RiskManager , market : pd.DataFrame):
        latest = market.iloc[-1]
        
        decisioncall = latest["ActionHat"] # 0 B 1 S 2 H

        action = ["BUY","SELL","HOLD"][decisioncall]

        #MVP is make orders and trade them. Sell immediately. Crude but will work.

        if action == "HOLD":
            # Hold for now.
            return {"action" : action , "status" : "no operational change"}
            
        elif action == "SELL":
            #Sell everything and minimize exposure , keep EOD empty or small.
            # RM not gating sell lotsize right now
            # if currentAmount 0:
            # return something
            # elif currentAmount > 0:

            accountdetailsurl = self.platform.make_url(self.mode , "account",1)
            accountdetails = requests.get(url = accountdetailsurl)
            #make robust later
            account = accountdetails.json()["account"]
            positions = account["positions"]
            fxidx = [p["instrument"] for p in positions].index(self.instrument)
            pointedposition = positions[fxidx]
            currentLong = float(pointedposition["long"]["units"])

            if currentLong > 0:
                static_lotsize = 1005

                (minsize , maxsize , bestsize) = manager.gate_lotsize()
                
                if manager.insiderange(minsize , maxsize, static_lotsize):
                    lot_size = static_lotsize
                else: # outside range for dynamic
                    lot_size = bestsize


                # Extra Long not terrible at small size, 1e-6 for smoothness
                if currentLong + 1e-6 < lot_size:
                    return {"action" : action , "status" : "stock less than static budget, wait"}

                order = self.make_order(side = -1 , lot_size= lot_size , reduce_only = True)

                if order["status"] == "filled":
                    return order
            else:
                return {"action" : action , "status" : "no stock, wait for BUY"}
            
        else: # action == "BUY"
            # Make a single order.
            static_lotsize = 1005

            (minsize , maxsize , bestsize) = manager.gate_lotsize()

            if manager.insiderange(minsize , maxsize, static_lotsize):
                lot_size = static_lotsize
            else: # outside range for dynamic
                lot_size = bestsize

            order = self.make_order(side = 1 , lot_size= lot_size , reduce_only = False)

            if order["status"] == "filled": 
                return order

# Risk Manager that decides who and when (ignition) and where (vendor and platform).
# rm caps values like pnl and how much notional and how much leverage but 
# the value itself is decided by dynamic information based on the regime. 
# trader makes only data driven decisions
# regime provides the information. if bearish low leverage and one medium for example.
class RiskManager:
    def __init__(self , name : str , vendor : DataVendor , platform : Platform):
        self.name = name
        self.team = None
        self.vendor = vendor
        self.platform = platform

    def in_budget(self): 
                # Roughly 15 trades/day per trader budget
                # M1-M10 rough horizon gives 30-40 daily trades, to be found.
                # Only upswings on buy and sell only means roughly half of that.
                # Set max trades daily per trader below.
                MAX_TRADES_PER_TRADER = 15
                STATIC_LOTSIZE = 1000
                
                # Query closed trades today (for whole account/team)
                running_trades = 0
                for trader in self.team:
                    running_trades += trader.platform.count_trades(trader.mode , trader.instrument)

                spent_today = running_trades*STATIC_LOTSIZE
                
                # Budget scales with team size
                budget = MAX_TRADES_PER_TRADER * len(self.team)*STATIC_LOTSIZE
                
                return spent_today < budget # trades today < total trades on the team daily
                # spent today can be made dynamic

    def in_bhe_hours(self,now_local: datetime | None = None) -> bool:
        """True iff Mon–Fri, 06:55–15:00 (America/Chicago)."""

        CHI = ZoneInfo("America/Chicago")
        START = time(6, 55)   # 06:55 AM CST/CDT
        END   = time(15, 0)   # 03:00 PM CST/CDT

        if now_local is None:
            now_local = datetime.now(CHI)
        # weekend: Saturday=5, Sunday=6
        if now_local.weekday() >= 5:
            return False
        t = now_local.time()
        return START <= t < END
        

    def ignition(self):
        #MVP involves a static ignition which comes down to
        # "Is Black Hole Enterprises Open" and "Are we under budget"
        if self.in_bhe_hours() and self.in_budget(): # Currently no PAUSED option, like in event of 
            # small blip but not catastrophe. Not implemented
            print("Arthur is open and trading...")
            # move the text to total ignition on arthur
            return "OPEN"
        else:
            print("No more trading for today.") #Make reason for no trading dynamic logging
            # move the text to total ignition on arthur
            return "CLOSED"
        
        
    def set_trader_team(self , team : list):
        self.team = team

    def insiderange(self , start : float , end : float , val : float) -> bool:
        return (start <= val <= end)

    
    def gate_lotsize(self):
        #Currently very static but becomes dynamic. 
        lotsizemin = 1000 
        lotsizemax = 2000
        lotsizebest = 1000 # gate functions give a range and a best value
        # and the best is not always a min or a max but is always within range

        return (lotsizemin , lotsizemax , lotsizebest)
    
    
    # RM has the key on all trading on "when business open" and
    # gates parameters. More gates coming. Gates minimize risk
    # whatever that looks like but traders are the dynamic ones. 
    # RM are controllers.

# Trading Platform
class Platform: # Generic Platform containing {x_j} None collection of all 
    def __init__(self): # possible base and endpoints. Specific Platforms use subcollection.
        self.authorization = None
        self.base = None
        self.endpoints = None

    def make_url(self , mode : str, endpoint: str , subEndpoint: int, **kw):
        # make_url generically uses a base, endpoint, and subEndpoint. 
        # Specifically, the context is the parameters used for the subEndpoint
        # and headers and params are added afterwards.
        # make_url lets all of the url be a string non formatted.

        context = {}

        context.update(**kw)

        # Platform make_url() uses actionType and does nothing with it because Platform doesn't exist.
        # Each subPlatform will 
        return self.base[mode] + self.endpoints[endpoint][subEndpoint].format(**context)

# Oanda Trading Platform (different from Oanda as Data Vendor)
# Currently OandaTrading is going to do both data and trading until I 
# distance the two and 'tie shoelaces' with Arthur, making Arthur modular
# with the global system. Might as well use Oanda for now, maybe interchange later.
class OandaTrading(Platform): # Specific Platform with slightly crossed wires
    def __init__(self, api_key : str): # and as normal as dynamic as possible with static info elsewhere.
        self.authorization = api_key
        self.accountID = None # Point to Nothing , Set to Anything
        self.instrument = None # Point to Nothing, Set to Anything
        
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
            "demo" : demo_base,
            "live" : live_base
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


    # Setters
    def set_accountID(self , accountID : str): self.accountID = accountID
    def set_instrument(self, instrument : str): self.instrument = instrument
    def set_orderSpecifier(self , orderSpecifier: str): self.orderSpecifier = orderSpecifier
    def set_tradeSpecifier(self , tradeSpecifier: str): self.tradeSpecifier = tradeSpecifier


    def make_url(self , mode : str, endpoint: str , subEndpoint: int , **kw):
        
        context = {
            "accountID" : self.accountID,
            "instrument": self.instrument,
            "orderSpecifier" : self.orderSpecifier,
            "tradeSpecifier" : self.tradeSpecifier
        }

        context.update(kw) # Updates context so that platform can stay unchanged and 
        # one can point to a different context rather than changing everything.

        return self.base[mode] + self.endpoints[endpoint][subEndpoint].format(**context)
    
    def close_trade(self , mode , trade_id):

        url = self.make_url(mode , "trade" , 3, tradeSpecifier = trade_id)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.authorization}",
            "Accept-Datetime-Format" : "RFC3339"
        }
         
        
        body = {
                "units" : "ALL" # Make it simple and close everything when possible.
                # Make more dynamic LATER
        }
    
        response = requests.put(url, headers = headers , json = body)
        
        if response.status_code == 200:
            return response.json() # finish or adapt, currently full body and can access all info
        else:
            raise ValueError(f"Error {response.status_code}: {response.text}")

    def cancel_order(self , mode , order_id):
        #see trader for details on why blank
        return None

    def make_order(self, mode , instrument : str , side : int, lot_size : int , reduce_only : bool = False):

        url = self.make_url(mode,"order",0)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.authorization}",
            "Accept-Datetime-Format" : "RFC3339"
        }
        
        
        body = {
                    "order": {
                        "units": str(side*lot_size),
                        "instrument": instrument,
                        "timeInForce": "FOK",
                        "type": "MARKET",
                        "positionFill": "REDUCE_ONLY" if reduce_only else "DEFAULT"
                    }
                }

        response = requests.post(url, headers = headers , json = body)
        
        if response.status_code == 201:
            return response.json() # finish or adapt, currently full body and can access all info
        else:
            raise ValueError(f"Error {response.status_code}: {response.text}")
        


    def count_trades(self , mode , instrument : str):
        
        url = self.make_url(mode , "trade", 0)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.authorization}",
            "Accept-Datetime-Format" : "RFC3339"
        }

        params = {
            "state" : "CLOSED",
            "instrument" : instrument,
            "count" : 500 # Maximum allowed, filter manually by time, other platform might 
            # allow more but... what individual 'simulated trader' needs N > 500?
        }

        request_body = requests.get(url = url , headers = headers , params = params)

        if request_body.status_code == 200:
            last_count_trades = request_body.json()["trades"]
        else:
            raise ValueError(f"Error {request_body.status_code} {request_body.text}")
        
        chi = ZoneInfo("America/Chicago")
        now_chi = datetime.now(chi)
        midnight_chi = datetime(year = now_chi.year , month = now_chi.month , day = now_chi.day , tzinfo = ZoneInfo("America/Chicago"))

        def to_chi(rfc3339_utc: str) -> datetime:
            # handle nanoseconds: split off 'Z', then split fractional seconds
            base = rfc3339_utc.rstrip("Z")
            if "." in base:
                base = base.split(".")[0]
            dt_utc = datetime.fromisoformat(base).replace(tzinfo=ZoneInfo("UTC"))
            return dt_utc.astimezone(chi)

        # local filter: today, optionally by instrument
        count = 0
        for t in last_count_trades:
            ct = t.get("closeTime")
            if not ct:
                continue
            ct_chi = to_chi(ct)
            if midnight_chi <= ct_chi <= now_chi:
                count += 1

        return count