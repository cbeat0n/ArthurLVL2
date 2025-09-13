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


# Data Vendor (vendor, bank, other digital ClientServer streams)
class DataVendor: # Generic Data Vendor containing {x_j} None collection of all 
    def __init__(self): # possible base and endpoints. Specific Data Vendors use subcollection.
        self.base = None
        self.endpoints = None
# Trader(s) that decides what and when (timing) based on instrument and regime from vendor
# trade function make it static with dynamic lot size
class Trader:
    def __init__(self):
        self.mode = None
        self.instrument = None
        self.intakeGranularity = None
        self.horizonGranularity = None
        self.tradingPlatform = None
        
    def close_trade(self ,  trade_id):
        # close_trade function partially inherited based on platform
        return self.tradingPlatform.close_trade(self.mode , trade_id)

    def cancel_order(self , order_id):
        # cancel_order function partially inherited based on platform
        return self.tradingPlatform.cancel_order(self.mode , order_id)


    def make_order(self , instrument : str , side , lot_size : float):
        # cancel_order function partially inherited based on platform
        return self.tradingPlatform.make_order(self.mode , instrument , side , lot_size)
    

# Risk Manager that decides who and when (ignition) and where (vendor and platform).
# rm caps values like pnl and how much notional and how much leverage but 
# the value itself is decided by dynamic information based on the regime. 
# trader makes only data driven decisions
# regime provides the information. if bearish low leverage and one medium for example.

# Trading Platform
class Platform: # Generic Platform containing {x_j} None collection of all 
    def __init__(self): # possible base and endpoints. Specific Platforms use subcollection.
        self.base = None
        self.endpoints = None

    def make_url(self , mode : str, endpoint: str , subEndpoint: int, actionType: str):
        # make_url generically uses a base, endpoint, and subEndpoint. 
        # Specifically, the context is the parameters used for the subEndpoint
        # and headers and params are added afterwards.
        # make_url lets all of the url be a string non formatted.

        # Platform make_url() uses actionType and does nothing with it because Platform doesn't exist.
        # Each subPlatform will 
        return self.base[mode] + self.endpoints[endpoint][subEndpoint]

# Oanda Trading Platform (different from Oanda as Data Vendor)
# Currently OandaTrading is going to do both data and trading until I 
# distance the two and 'tie shoelaces' with Arthur, making Arthur modular
# with the global system. Might as well use Oanda for now, maybe interchange later.
class OandaTrading(Platform): # Specific Platform with slightly crossed wires
    def __init__(self): # and as normal as dynamic as possible with static info elsewhere.
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
    
    def close_trade(mode , trade_id):
        return None

    def cancel_order(mode , order_id):
        return None

    def make_order(mode , instrument : str , side , lot_size : float):
        return None
    




