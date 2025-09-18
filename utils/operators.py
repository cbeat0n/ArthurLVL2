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
from datetime import datetime, time
from zoneinfo import ZoneInfo  # Python 3.9+

# Data Vendor (vendor, bank, other digital ClientServer streams)
class DataVendor: # Generic Data Vendor containing {x_j} None collection of all 
    def __init__(self): # possible base and endpoints. Specific Data Vendors use subcollection.
        self.base = None
        self.endpoints = None
# Trader(s) that decides what and when (timing) based on instrument and regime from vendor
# trade function make it static with dynamic lot size
class Trader:
    def __init__(self ,name : str, mode : str , instrument : str , platform : Platform , intake : object , horizon : object):
        self.name = name
        self.mode = mode
        self.instrument = instrument
        self.intakeGranularity = intake # move to datavendor
        self.horizonGranularity = horizon # move to datavendor
        self.tradingPlatform = platform
        
    def close_trade(self ,  trade_id):
        # close_trade function partially inherited based on platform

        closure = self.tradingPlatform.close_trade(self.mode , trade_id)

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

        return self.tradingPlatform.cancel_order(self.mode , order_id)


    def make_order(self , side , lot_size : float):
        # make_order function partially inherited based on platform

        order = self.tradingPlatform.make_order(self.mode , self.instrument , side , lot_size)

        if order.get("orderFillTransaction") :
            return {"status": "filled" , "raw" : order , 
                    "order_id" : order["orderFillTransaction"]["orderID"],
                    "trade_id" : order["orderFillTransaction"]["tradeOpened"]["tradeID"]}
        
        elif order.get("orderCancelTransaction"):
            return {"status" : "killed" , "raw" : order , "order_id" : order["orderCancelTransaction"]["orderID"]}
        
        
    

# Risk Manager that decides who and when (ignition) and where (vendor and platform).
# rm caps values like pnl and how much notional and how much leverage but 
# the value itself is decided by dynamic information based on the regime. 
# trader makes only data driven decisions
# regime provides the information. if bearish low leverage and one medium for example.
class RiskManager:
    def __init__(self , name : str , team : list , vendor : DataVendor , platform : Platform):
        self.name = name
        self.team = team
        self.vendor = vendor
        self.platform = platform

    def ignition(self):
        #MVP involves a static ignition which comes down to
        # "Is Black Hole Enterprises Open" and "Are we under budget"

        CHI = ZoneInfo("America/Chicago")
        START = time(6, 55)   # 06:55 AM CST/CDT
        END   = time(15, 0)   # 03:00 PM CST/CDT

        def in_bhe_hours(now_local: datetime | None = None) -> bool:
            """True iff Mon–Fri, 06:55–15:00 (America/Chicago)."""
            if now_local is None:
                now_local = datetime.now(CHI)
            # weekend: Saturday=5, Sunday=6
            if now_local.weekday() >= 5:
                return False
            t = now_local.time()
            return START <= t < END
        
        def in_budget(): 
            # Roughly 15 trades/day per trader budget
            # M1-M10 rough horizon gives 30-40 daily trades, to be found.
            # Only upswings on buy and sell only means roughly half of that.
            # Set max trades daily per trader below.
            MAX_TRADES_PER_TRADER = 15
            STATIC_LOTSIZE = 1000
            
            # Query closed trades today (for whole account/team)
            spent_today = self.platform.count_trades_today()*STATIC_LOTSIZE
            
            # Budget scales with team size
            budget = MAX_TRADES_PER_TRADER * len(self.team)*STATIC_LOTSIZE
            
            return spent_today < budget # trades today < total trades on the team daily
            # spent today can be made dynamic
        
        if not in_bhe_hours():
            print("You are not in Black Hole Enterprises business hours.")
            raise SystemExit(0)
        elif in_budget():
            return True
        else:
            #Ignition gates whether or not trading, but does not shut the system off unless
            # not INBHEHOURS.
            return False
        


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

    def make_order(self, mode , instrument : str , side : int, lot_size : int):

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
                        "positionFill": "DEFAULT"
                    }
                }

        response = requests.post(url, headers = headers , json = body)
        
        if response.status_code == 201:
            return response.json() # finish or adapt, currently full body and can access all info
        else:
            raise ValueError(f"Error {response.status_code}: {response.text}")
        




