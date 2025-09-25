### transform_auto.py is a modular and agnostic framework that 
### serves as the core engine for Arthur when it comes to trading.
### It is a part of proprietary ETL in ETL-Auto for Arthur.

## External Dependencies
import pandas as pd # Library for data manipulation
import numpy as np # Library for numerical calculations
import requests # Library for HTTP requests
from datetime import datetime , timezone # Library for time objects 

## Internal Dependencies
from utils.transform_signal import REGIME 
from weeklybacktesting.backtestingprocess import pip_size

## Columns inside the finalized matrix.
COLS = ["accountID","name" ,"riskmanager","ts", "d_int" , "d_hor" , "d_zone", "instrument" ,"regime" , "B", "S" ,"Margin","Intensity", "U" , "V" , "UxV", 
        "ActionHat", "lotsize_roll", "B_a" , "S_a" , "T_a" , "why_unfill" , "liquidity", "bid",
        "ask" , "spread" , "intended" , "fill_roll" , "slip_r_roll" , "slip_a_roll" , "slip_pips_roll" , "open", "high" , "low",
        "close" , "volume" , "open_orders" , "open_trades" , "p_roll" , "G_roll" , "L_roll",
        "N_B" , "N_S" , "N_T" , "C_roll" , "PnL"]



def finalize_records(signal : pd.DataFrame , trader, manager,  * , record = None , **kwargs):
    """
    finalize_records takes a variety of inputs performs proprietary ETL on the information presented by the 
    vendor ( or platform for MVP ). It simulates a sample measurement after every OODA LOOP trade.
    It returns a JSON containing the finalized record.

    Input:
    signal : pd.DataFrame - signal matrix incoming from ETL-Base
    trader : Trader - the trader observing and measuring
    manager : RiskManager - the trader's RiskManager
    ** Separated for optional arguments **
    record : None , pd.DataFrame - the record if it already exists
    **kwargs : a set of key value pairs known as keyword arguments
    """

    used_endpoints = {
        "t_o_e" : None, "t_o_s" : None, # today orders end or subend
        "t_t_e" : None, "t_t_s" : None, # today trades end or subend
        "pricing" : None, "pricing_sub" : None, # pricing end or subend
        "acc_e" : None, "acc_s" : None, # account end or subend
        "transaction_e" : None, "transaction_s" : None} # transaction end or subend

    used_endpoints.update(**kwargs) # kwargs are key value pairs rolled out, or deferred parameters

    ## Statistics are either live or rolling, labeled _roll if rolling.
    ## Live -> happening at intake frequency, like OHLCV
    ## Rolling -> happening more frequently, rolling already at intake, like G and L or C

    if record is None or (isinstance(record, pd.DataFrame) and record.empty):

        finalized_record = pd.DataFrame(columns = COLS)

        firstrow = {column : None for column in COLS}

        ## Current UTC
        now = datetime.now(timezone.utc)

        ## Midnight UTC today
        midnight_utc_today = datetime(year=now.year, month=now.month, day=now.day, tzinfo=timezone.utc)
        
        ## Find the orders and trades for today for the trader in question
        today_orders = trader.config.platform.make_url(trader.config.mode , used_endpoints["t_o_e"] , used_endpoints["t_o_s"] , accountID = trader.config.platform.accountID)
        today_trades = trader.config.platform.make_url(trader.config.mode , used_endpoints["t_t_e"] , used_endpoints["t_t_s"] , accountID = trader.config.platform.accountID)

        ## Find the orders that are filled
        today_f_orders = trader.config.session.get(url = today_orders , params = {"state" : "FILLED" , "count" : 200})

        if today_f_orders.status_code == 200:
            today_f_orders = today_f_orders.json()["orders"]
        else:
            raise ValueError(f"Error {today_f_orders.status_code}: {today_f_orders.text}")

        ## Find the trades that are closed
        today_c_trades = trader.config.session.get(url = today_trades ,  params = {"state" : "CLOSED" , "count" : 200})

        if today_c_trades.status_code == 200:
            today_c_trades = today_c_trades.json()["trades"]
        else:
            raise ValueError(f"Error {today_c_trades.status_code}: {today_c_trades.text}")


        ## Find pricing information on the instrument from vendor , currently platform because OANDA.
        pricing_url = trader.config.platform.make_url(trader.config.mode, used_endpoints["pricing"] , used_endpoints["pricing_sub"] , accountID = trader.config.platform.accountID)

        pricing = trader.config.session.get(url = pricing_url, params = {"instruments" : trader.config.instrument})

        if pricing.status_code == 200:
            pricing = pricing.json()["prices"]
        else:
            raise ValueError(f"Error {pricing.status_code}: {pricing.text}")
        
        ## Find account information on the trader in question.
        account_url = trader.config.platform.make_url(trader.config.mode , used_endpoints["acc_e"] , used_endpoints["acc_s"] , accountID = trader.config.platform.accountID)

        account = trader.config.session.get(url = account_url)

        if account.status_code == 200:
            account = account.json()["account"]
        else:
            raise ValueError(f"Error {account.status_code}: {account.text}")


        ## Find transactions today for codes for ETL
        transaction_url = trader.config.platform.make_url(trader.config.mode , used_endpoints["transaction_e"] , used_endpoints["transaction_s"] , accountID = trader.config.platform.accountID)

        transaction_body = trader.config.session.get(url = transaction_url , params = {"from" : midnight_utc_today.strftime("%Y-%m-%dT%H:%M:%SZ") , "to" : datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                      , "type" : "ORDER_FILL"})

        if transaction_body.status_code == 200:
            pages = transaction_body.json().get("pages" , [])
        else:
            raise ValueError(f"Error {transaction_body.status_code}: {transaction_body.text}")
        
        
        pages_url = None  # reset every time
        order_fills = [] # default safe value

        # Pick the first valid URL in pages
        if isinstance(pages, list):
            for candidate in pages:
                if isinstance(candidate, str) and candidate.startswith("http"):
                    pages_url = candidate
                    break

        if isinstance(pages_url, str):
            range_transactions = trader.config.session.get(url=pages_url)
            if range_transactions.status_code == 200:
                order_fills = range_transactions.json().get("transactions", [])
            else:
                raise ValueError(f"Error {range_transactions.status_code}: {range_transactions.text}")
        else:
            # Debugging breadcrumb so you can see what OANDA gave you
            print(f"[WARN] No valid pages returned from OANDA: {pages}")
            order_fills = []


        ## Intermediary calculations
        lotsize_roll = np.median([abs(int(t["units"]))
                for t in today_f_orders if t["instrument"] ==  trader.config.instrument
                and t["createTime"][0:10] == signal["timestamp"][0][0:10]]) if any(today_f_orders) else 0.0 # closed only

        
        liquidity = pricing[0]["asks"][0]["liquidity"]
        
        bid = float(pricing[0]["bids"][0]["price"])
        ask = float(pricing[0]["asks"][0]["price"])

        fills = [float(order["price"]) for order in order_fills if order["instrument"] == trader.config.instrument]
        fill_roll = np.median(fills) if any(fills) else 0.0


        G = sum(1 for t in today_c_trades if float(t["realizedPL"]) > 0)
        L = sum(1 for t in today_c_trades if float(t["realizedPL"]) < 0)
        
        p_roll = G / (G+L) if (G+L) else 0.0
        
        Gs = [float(t["realizedPL"]) if float(t["realizedPL"]) > 0 else 0.0 for t in today_c_trades]
        Ls = [abs(float(t["realizedPL"])) if float(t["realizedPL"]) < 0 else 0.0 for t in today_c_trades]
        
        G_roll = np.median(Gs) if any(Gs) else 0.0
        L_roll = np.median(Ls) if any(Ls) else 0.0
        
        num_buy = sum(1 for t in today_c_trades if float(t["initialUnits"]) > 0)
        num_sell = sum(1 for t in today_c_trades if float(t["initialUnits"]) < 0)

        open_orders = account["pendingOrderCount"]
        open_trades = account["openTradeCount"]
        
        ## PnL might not exist for given asset being retrieved , else PnL is realized and unrealized
        pnl = 0.0
        for pos in account.get("positions", []):
            if pos.get("instrument") == trader.config.instrument:
                pnl = float(pos.get("pl", 0.0)) + float(pos.get("unrealizedPL", 0.0))
                break


        ## Make a JSON of the row to be C, so that finalized matrix has U in CRUD.
        firstrow["accountID"] = trader.config.platform.accountID
        firstrow["name"] = trader.name
        firstrow["riskmanager"] = manager.name
        firstrow["ts"] = signal["timestamp"].iloc[-1]
        firstrow["d_int"] = trader.config.intake
        firstrow["d_hor"] = trader.config.horizon
        firstrow["d_zone"] = trader.config.zone
        firstrow["instrument"] = trader.config.instrument
        firstrow["regime"] = REGIME
        firstrow["B"] = signal["B"]
        firstrow["S"] = signal["S"]
        firstrow["Margin"] = signal["Margin"]
        firstrow["Intensity"] = signal["Intensity"]
        firstrow["U"] = signal["Theta"]
        firstrow["V"] = signal["Mu"]
        firstrow["UxV"] = "Theta x Mu" #dynamically pull if more fluid else hardcode
        firstrow["ActionHat"] = signal["ActionHat"]
        firstrow["lotsize_roll"] = lotsize_roll
        firstrow["B_a"] = None #B_a with dynamic filling later, FOK
        firstrow["S_a"] = None #S_a with dynamic filling later, FOK
        firstrow["T_a"] = None #T_a with dynamic filling later, FOK
        firstrow["why_unfill"] = None #why_unfill with dynamic filling later, FOK
        firstrow["liquidity"] = liquidity
        firstrow["bid"] = bid
        firstrow["ask"] = ask
        firstrow["spread"] = ask - bid
        firstrow["intended"] = ask if signal["ActionHat"].iloc[-1] == 0 else (bid if signal["ActionHat"].iloc[-1] == 1 else 0)
        firstrow["fill_roll"] = fill_roll
        firstrow["slip_r_roll"] = fill_roll - firstrow["intended"]
        firstrow["slip_a_roll"] = np.abs(firstrow["slip_r_roll"])
        firstrow["slip_pips_roll"] = firstrow["slip_a_roll"] / pip_size(firstrow["instrument"])
        firstrow["open"] = signal["open"] # ohlcv needs to be in signal, tune the t 2
        firstrow["high"] = signal["high"]
        firstrow["low"] = signal["low"] 
        firstrow["close"] = signal["close"]
        firstrow["volume"] = signal["volume"]
        firstrow["open_orders"] = open_orders # call pending orders from account but 0 not useful rn
        firstrow["open_trades"] = open_trades
        firstrow["p_roll"] = p_roll
        firstrow["G_roll"] = G_roll
        firstrow["L_roll"] = L_roll
        firstrow["N_B"] = num_buy
        firstrow["N_S"] = num_sell
        firstrow["N_T"] = firstrow["N_B"] + firstrow["N_S"]
        firstrow["C_roll"] = None #C_roll for later b/c vendor hide. PnL given
        firstrow["PnL"] = pnl

        finalized_record.loc[len(finalized_record)] = firstrow

        return {"finalrecord" : finalized_record}
    
    elif isinstance(record, pd.DataFrame) and not record.empty:
        
        finalized_record = record

        firstrow = {column : None for column in COLS}

        ## Current UTC
        now = datetime.now(timezone.utc)

        ## Midnight UTC today
        midnight_utc_today = datetime(year=now.year, month=now.month, day=now.day, tzinfo=timezone.utc)
        
        ## Find the orders and trades for today for the trader in question
        today_orders = trader.config.platform.make_url(trader.config.mode , used_endpoints["t_o_e"] , used_endpoints["t_o_s"] , accountID = trader.config.platform.accountID)
        today_trades = trader.config.platform.make_url(trader.config.mode , used_endpoints["t_t_e"] , used_endpoints["t_t_s"] , accountID = trader.config.platform.accountID)

        ## Find the orders that are filled
        today_f_orders = trader.config.session.get(url = today_orders , params = {"state" : "FILLED" , "count" : 200})

        if today_f_orders.status_code == 200:
            today_f_orders = today_f_orders.json()["orders"]
        else:
            raise ValueError(f"Error {today_f_orders.status_code}: {today_f_orders.text}")

        ## Find the trades that are closed
        today_c_trades = trader.config.session.get(url = today_trades ,  params = {"state" : "CLOSED" , "count" : 200})

        if today_c_trades.status_code == 200:
            today_c_trades = today_c_trades.json()["trades"]
        else:
            raise ValueError(f"Error {today_c_trades.status_code}: {today_c_trades.text}")


        ## Find pricing information on the instrument from vendor , currently platform because OANDA.
        pricing_url = trader.config.platform.make_url(trader.config.mode, used_endpoints["pricing"] , used_endpoints["pricing_sub"] , accountID = trader.config.platform.accountID)

        pricing = trader.config.session.get(url = pricing_url, params = {"instruments" : trader.config.instrument})

        if pricing.status_code == 200:
            pricing = pricing.json()["prices"]
        else:
            raise ValueError(f"Error {pricing.status_code}: {pricing.text}")
        
        ## Find account information on the trader in question.
        account_url = trader.config.platform.make_url(trader.config.mode , used_endpoints["acc_e"] , used_endpoints["acc_s"] , accountID = trader.config.platform.accountID)

        account = trader.config.session.get(url = account_url)

        if account.status_code == 200:
            account = account.json()["account"]
        else:
            raise ValueError(f"Error {account.status_code}: {account.text}")


        ## Find transactions today for codes for ETL
        transaction_url = trader.config.platform.make_url(trader.config.mode , used_endpoints["transaction_e"] , used_endpoints["transaction_s"] , accountID = trader.config.platform.accountID)

        transaction_body = trader.config.session.get(url = transaction_url , params = {"from" : midnight_utc_today.strftime("%Y-%m-%dT%H:%M:%SZ") , "to" : datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                      , "type" : "ORDER_FILL"})

        if transaction_body.status_code == 200:
            pages = transaction_body.json().get("pages" , [])
        else:
            raise ValueError(f"Error {transaction_body.status_code}: {transaction_body.text}")
        
        
        pages_url = None  # reset every time
        order_fills = [] # default safe value

        # Pick the first valid URL in pages
        if isinstance(pages, list):
            for candidate in pages:
                if isinstance(candidate, str) and candidate.startswith("http"):
                    pages_url = candidate
                    break

        if isinstance(pages_url, str):
            range_transactions = trader.config.session.get(url=pages_url)
            if range_transactions.status_code == 200:
                order_fills = range_transactions.json().get("transactions", [])
            else:
                raise ValueError(f"Error {range_transactions.status_code}: {range_transactions.text}")
        else:
            # Debugging breadcrumb so you can see what OANDA gave you
            print(f"[WARN] No valid pages returned from OANDA: {pages}")
            order_fills = []


        ## Intermediary calculations
        lotsize_roll = np.median([abs(int(t["units"]))
                for t in today_f_orders if t["instrument"] ==  trader.config.instrument
                and t["createTime"][0:10] == signal["timestamp"][0][0:10]]) if any(today_f_orders) else 0.0 # closed only

        
        liquidity = pricing[0]["asks"][0]["liquidity"]
        
        bid = float(pricing[0]["bids"][0]["price"])
        ask = float(pricing[0]["asks"][0]["price"])

        fills = [float(order["price"]) for order in order_fills if order["instrument"] == trader.config.instrument]
        fill_roll = np.median(fills) if any(fills) else 0.0


        G = sum(1 for t in today_c_trades if float(t["realizedPL"]) > 0)
        L = sum(1 for t in today_c_trades if float(t["realizedPL"]) < 0)
        
        p_roll = G / (G+L) if (G+L) else 0.0
        
        Gs = [float(t["realizedPL"]) if float(t["realizedPL"]) > 0 else 0.0 for t in today_c_trades]
        Ls = [abs(float(t["realizedPL"])) if float(t["realizedPL"]) < 0 else 0.0 for t in today_c_trades]
        
        G_roll = np.median(Gs) if any(Gs) else 0.0
        L_roll = np.median(Ls) if any(Ls) else 0.0
        
        num_buy = sum(1 for t in today_c_trades if float(t["initialUnits"]) > 0)
        num_sell = sum(1 for t in today_c_trades if float(t["initialUnits"]) < 0)

        open_orders = account["pendingOrderCount"]
        open_trades = account["openTradeCount"]
        
        ## PnL might not exist for given asset being retrieved , else PnL is realized and unrealized
        pnl = 0.0
        for pos in account.get("positions", []):
            if pos.get("instrument") == trader.config.instrument:
                pnl = float(pos.get("pl", 0.0)) + float(pos.get("unrealizedPL", 0.0))
                break


        ## Make a JSON of the row to be C, so that finalized matrix has U in CRUD.
        firstrow["accountID"] = trader.config.platform.accountID
        firstrow["name"] = trader.name
        firstrow["riskmanager"] = manager.name
        firstrow["ts"] = signal["timestamp"].iloc[-1]
        firstrow["d_int"] = trader.config.intake
        firstrow["d_hor"] = trader.config.horizon
        firstrow["d_zone"] = trader.config.zone
        firstrow["instrument"] = trader.config.instrument
        firstrow["regime"] = REGIME
        firstrow["B"] = signal["B"]
        firstrow["S"] = signal["S"]
        firstrow["Margin"] = signal["Margin"]
        firstrow["Intensity"] = signal["Intensity"]
        firstrow["U"] = signal["Theta"]
        firstrow["V"] = signal["Mu"]
        firstrow["UxV"] = "Theta x Mu" #dynamically pull if more fluid else hardcode
        firstrow["ActionHat"] = signal["ActionHat"]
        firstrow["lotsize_roll"] = lotsize_roll
        firstrow["B_a"] = None #B_a with dynamic filling later, FOK
        firstrow["S_a"] = None #S_a with dynamic filling later, FOK
        firstrow["T_a"] = None #T_a with dynamic filling later, FOK
        firstrow["why_unfill"] = None #why_unfill with dynamic filling later, FOK
        firstrow["liquidity"] = liquidity
        firstrow["bid"] = bid
        firstrow["ask"] = ask
        firstrow["spread"] = ask - bid
        firstrow["intended"] = ask if signal["ActionHat"].iloc[-1] == 0 else (bid if signal["ActionHat"].iloc[-1] == 1 else 0)
        firstrow["fill_roll"] = fill_roll
        firstrow["slip_r_roll"] = fill_roll - firstrow["intended"]
        firstrow["slip_a_roll"] = np.abs(firstrow["slip_r_roll"])
        firstrow["slip_pips_roll"] = firstrow["slip_a_roll"] / pip_size(firstrow["instrument"])
        firstrow["open"] = signal["open"] # ohlcv needs to be in signal, tune the t 2
        firstrow["high"] = signal["high"]
        firstrow["low"] = signal["low"] 
        firstrow["close"] = signal["close"]
        firstrow["volume"] = signal["volume"]
        firstrow["open_orders"] = open_orders # call pending orders from account but 0 not useful rn
        firstrow["open_trades"] = open_trades
        firstrow["p_roll"] = p_roll
        firstrow["G_roll"] = G_roll
        firstrow["L_roll"] = L_roll
        firstrow["N_B"] = num_buy
        firstrow["N_S"] = num_sell
        firstrow["N_T"] = firstrow["N_B"] + firstrow["N_S"]
        firstrow["C_roll"] = None #C_roll for later b/c vendor hide. PnL given
        firstrow["PnL"] = pnl

        finalized_record.loc[len(finalized_record)] = firstrow

        return {"finalrecord" : finalized_record}