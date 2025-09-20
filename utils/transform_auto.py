### transform_auto.py is a modular and agnostic framework that trades for Arthur
### and constructs the finalized matrix. It returns the filepath as well as the matrix
### itself.

# Import Dependencies
import pandas as pd
import numpy as np
import requests
from datetime import datetime , timezone
from utils.generic import Trader , RiskManager
from utils.transform_signal import REGIME
from verificationlearning.historicaldataverification import pip_size

# modular and agnostic of any trade. scrapes rolling PnL et al info from platform.
COLS = ["accountID","name" ,"riskmanager","ts", "d_int" , "d_hor" , "instrument" ,"regime" , "B", "S" ,"Margin", "U" , "V" , "UxV", 
        "ActionHat", "lotsize_roll", "B_a" , "S_a" , "T_a" , "why_unfill" , "liquidity", "bid",
        "ask" , "spread" , "intended" , "fill_roll" , "slip_r_roll" , "slip_a_roll" , "slip_pips_roll" , "open", "high" , "low",
        "close" , "volume" , "open_orders" , "open_trades" , "p_roll" , "G_roll" , "L_roll",
        "N_B" , "N_S" , "N_T" , "C_roll" , "PnL"]



def finalize_records(signal : pd.DataFrame , trader : Trader , manager : RiskManager,  * , record = None , **kwargs):

    """
    Big comments can go back here after the MVP is done besides tuning.
    """

    used_endpoints = {
        "t_o_e" : None,
        "t_o_s" : None,
        "t_t_e" : None,
        "t_t_s" : None,
        "pricing" : None,
        "pricing_sub" : None,
        "acc_e" : None,
        "acc_s" : None,
        "transaction_e" : None,
        "transaction_s" : None
    }

    used_endpoints.update(**kwargs)

    ## Statistics are either live or rolling, labeled _roll if rolling.
    # Live -> happening at intake frequency, like OHLCV
    # Rolling -> happening more frequently, rolling already at intake, like G and L or C

    if record == None:

        finalized_record = pd.DataFrame(columns = COLS)

        firstrow = {column : None for column in COLS}

        # current UTC
        now = datetime.now(timezone.utc)

        # midnight UTC today
        midnight_utc_today = datetime(year=now.year, month=now.month, day=now.day, tzinfo=timezone.utc)
        
        today_orders = trader.platform.make_url(trader.mode , used_endpoints["t_o_e"] , used_endpoints["t_o_s"] , accountID = trader.platform.accountID)
        today_trades = trader.platform.make_url(trader.mode , used_endpoints["t_t_e"] , used_endpoints["t_t_s"] , accountID = trader.platform.accountID)


        today_f_orders = requests.get(url = today_orders , headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {trader.platform.authorization}",
            "Accept-Datetime-Format" : "RFC3339"
        } , params = {"state" : "FILLED" , "count" : 200})

        if today_f_orders.status_code == 200:
            today_f_orders = today_f_orders.json()["orders"]
        else:
            raise ValueError(f"Error {today_f_orders.status_code}: {today_f_orders.text}")

        today_c_trades = requests.get(url = today_trades , headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {trader.platform.authorization}",
            "Accept-Datetime-Format" : "RFC3339"
        }   ,  params = {"state" : "CLOSED" , "count" : 200})

        if today_c_trades.status_code == 200:
            today_c_trades = today_c_trades.json()["trades"]
        else:
            raise ValueError(f"Error {today_c_trades.status_code}: {today_c_trades.text}")




        pricing_url = trader.platform.make_url(trader.mode, used_endpoints["pricing"] , used_endpoints["pricing_sub"] , accountID = trader.platform.accountID)

        pricing = requests.get(url = pricing_url , headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {trader.platform.authorization}",
            "Accept-Datetime-Format" : "RFC3339"
        } , params = {"instruments" : trader.instrument})


        if pricing.status_code == 200:
            pricing = pricing.json()["prices"]
        else:
            raise ValueError(f"Error {pricing.status_code}: {pricing.text}")
        

        transaction_url = trader.platform.make_url(trader.mode , used_endpoints["transaction_e"] , used_endpoints["transaction_s"] , accountID = trader.platform.accountID)

        transaction_body = requests.get(url = transaction_url , headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {trader.platform.authorization}",
            "Accept-Datetime-Format" : "RFC3339"
        } , params = {"from" : midnight_utc_today.strftime("%Y-%m-%dT%H:%M:%SZ") , "to" : datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                      , "type" : "ORDER_FILL"})

        if transaction_body.status_code == 200:
            pages_url = transaction_body.json()["pages"][-1]
        else:
            raise ValueError(f"Error {transaction_body.status_code}: {transaction_body.text}")
        
        range_transactions = requests.get(url = pages_url , headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {trader.platform.authorization}",
            "Accept-Datetime-Format" : "RFC3339"
        })

        if range_transactions.status_code == 200:
            order_fills = range_transactions.json()["transactions"]
        else:
            raise ValueError(f"Error {range_transactions.status_code}: {range_transactions.text}")

        account_url = trader.platform.make_url(trader.mode , used_endpoints["acc_e"] , used_endpoints["acc_s"] , accountID = trader.platform.accountID)

        account = requests.get(url = account_url , headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {trader.platform.authorization}",
            "Accept-Datetime-Format" : "RFC3339"
        })

        if account.status_code == 200:
            account = account.json()["account"]
        else:
            raise ValueError(f"Error {account.status_code}: {account.text}")

        lotsize_roll = np.median([abs(int(t["units"]))
                        for t in today_f_orders if t["instrument"] ==  trader.instrument
                        and t["createTime"][0:10] == signal["timestamp"][0][0:10]]) if any(today_f_orders) else 0.0 # closed only
        
        
        
        liquidity = pricing[0]["asks"][0]["liquidity"]
        bid = float(pricing[0]["bids"][0]["price"])
        ask = float(pricing[0]["asks"][0]["price"])

        fills = [float(order["price"]) for order in order_fills if order["instrument"] == trader.instrument ]
        fill_roll = np.median(fills) if any(fills) else 0.0


        G = sum(1 for t in today_c_trades if float(t["realizedPL"]) > 0)
        L = sum(1 for t in today_c_trades if float(t["realizedPL"]) < 0)
        p_roll = G / (G+L) if (G+L) else 0.0
        Gs = [float(t["realizedPL"]) if float(t["realizedPL"]) > 0 else 0.0 for t in today_c_trades]
        Ls = [float(t["realizedPL"]) if float(t["realizedPL"]) < 0 else 0.0 for t in today_c_trades]
        G_roll = np.median(Gs) if any(Gs) else 0.0
        L_roll = np.median(Ls) if any(Ls) else 0.0
        num_buy = sum(1 for t in today_c_trades if float(t["initialUnits"]) > 0)
        num_sell = sum(1 for t in today_c_trades if float(t["initialUnits"]) < 0)

        open_orders = account["pendingOrderCount"]
        open_trades = account["openTradeCount"]
        fxidx = [p["instrument"] for p in account["positions"]].index(trader.instrument)
        pnl = float(account["positions"][fxidx][trader.instrument]["pl"]) + float(account["positions"][fxidx][trader.instrument]["unrealizedPL"])



        
        firstrow["accountID"] = trader.platform.accountID
        firstrow["name"] = trader.name
        firstrow["riskmanager"] = manager.name
        firstrow["ts"] = signal["timestamp"][-1]
        firstrow["d_int"] = trader.intakeGranularity
        firstrow["d_hor"] = trader.horizonGranularity
        firstrow["instrument"] = trader.instrument
        firstrow["regime"] = REGIME
        firstrow["B"] = signal["B"]
        firstrow["S"] = signal["S"]
        firstrow["Margin"] = signal["Margin"]
        firstrow["U"] = signal["theta"]
        firstrow["V"] = signal["mu"]
        firstrow["UxV"] = "theta x mu" #dynamically pull if more fluid else hardcode
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
        firstrow["intended"] = [ask if signal["ActionHat"] == 0 else (bid if signal["ActionHat"] == 1 else 0)][0]
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

        return finalized_record
    
    elif len(record) > 0:
        
        finalized_record = record

        firstrow = {column : None for column in COLS}

        # current UTC
        now = datetime.now(timezone.utc)

        # midnight UTC today
        midnight_utc_today = datetime(year=now.year, month=now.month, day=now.day, tzinfo=timezone.utc)
        
        today_orders = trader.platform.make_url(trader.mode , used_endpoints["t_o_e"] , used_endpoints["t_o_s"] , accountID = trader.platform.accountID)
        today_trades = trader.platform.make_url(trader.mode , used_endpoints["t_t_e"] , used_endpoints["t_t_s"] , accountID = trader.platform.accountID)


        today_f_orders = requests.get(url = today_orders , headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {trader.platform.authorization}",
            "Accept-Datetime-Format" : "RFC3339"
        } , params = {"state" : "FILLED" , "count" : 200})

        if today_f_orders.status_code == 200:
            today_f_orders = today_f_orders.json()["orders"]
        else:
            raise ValueError(f"Error {today_f_orders.status_code}: {today_f_orders.text}")

        today_c_trades = requests.get(url = today_trades , headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {trader.platform.authorization}",
            "Accept-Datetime-Format" : "RFC3339"
        }   ,  params = {"state" : "CLOSED" , "count" : 200})

        if today_c_trades.status_code == 200:
            today_c_trades = today_c_trades.json()["trades"]
        else:
            raise ValueError(f"Error {today_c_trades.status_code}: {today_c_trades.text}")




        pricing_url = trader.platform.make_url(trader.mode, used_endpoints["pricing"] , used_endpoints["pricing_sub"] , accountID = trader.platform.accountID)

        pricing = requests.get(url = pricing_url , headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {trader.platform.authorization}",
            "Accept-Datetime-Format" : "RFC3339"
        } , params = {"instruments" : trader.instrument})


        if pricing.status_code == 200:
            pricing = pricing.json()["prices"]
        else:
            raise ValueError(f"Error {pricing.status_code}: {pricing.text}")
        

        transaction_url = trader.platform.make_url(trader.mode , used_endpoints["transaction_e"] , used_endpoints["transaction_s"] , accountID = trader.platform.accountID)

        transaction_body = requests.get(url = transaction_url , headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {trader.platform.authorization}",
            "Accept-Datetime-Format" : "RFC3339"
        } , params = {"from" : midnight_utc_today.strftime("%Y-%m-%dT%H:%M:%SZ") , "to" : datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                      , "type" : "ORDER_FILL"})

        if transaction_body.status_code == 200:
            pages_url = transaction_body.json()["pages"][-1]
        else:
            raise ValueError(f"Error {transaction_body.status_code}: {transaction_body.text}")
        
        range_transactions = requests.get(url = pages_url , headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {trader.platform.authorization}",
            "Accept-Datetime-Format" : "RFC3339"
        })

        if range_transactions.status_code == 200:
            order_fills = range_transactions.json()["transactions"]
        else:
            raise ValueError(f"Error {range_transactions.status_code}: {range_transactions.text}")

        account_url = trader.platform.make_url(trader.mode , used_endpoints["acc_e"] , used_endpoints["acc_s"] , accountID = trader.platform.accountID)

        account = requests.get(url = account_url , headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {trader.platform.authorization}",
            "Accept-Datetime-Format" : "RFC3339"
        })

        if account.status_code == 200:
            account = account.json()["account"]
        else:
            raise ValueError(f"Error {account.status_code}: {account.text}")

        lotsize_roll = np.median([abs(int(t["units"]))
                        for t in today_f_orders if t["instrument"] ==  trader.instrument
                        and t["createTime"][0:10] == signal["timestamp"][0][0:10]]) if any(today_f_orders) else 0.0 # closed only
        
        
        
        liquidity = pricing[0]["asks"][0]["liquidity"]
        bid = float(pricing[0]["bids"][0]["price"])
        ask = float(pricing[0]["asks"][0]["price"])

        fills = [float(order["price"]) for order in order_fills if order["instrument"] == trader.instrument ]
        fill_roll = np.median(fills) if any(fills) else 0.0


        G = sum(1 for t in today_c_trades if float(t["realizedPL"]) > 0)
        L = sum(1 for t in today_c_trades if float(t["realizedPL"]) < 0)
        p_roll = G / (G+L) if (G+L) else 0.0
        Gs = [float(t["realizedPL"]) if float(t["realizedPL"]) > 0 else 0.0 for t in today_c_trades]
        Ls = [float(t["realizedPL"]) if float(t["realizedPL"]) < 0 else 0.0 for t in today_c_trades]
        G_roll = np.median(Gs) if any(Gs) else 0.0
        L_roll = np.median(Ls) if any(Ls) else 0.0
        num_buy = sum(1 for t in today_c_trades if float(t["initialUnits"]) > 0)
        num_sell = sum(1 for t in today_c_trades if float(t["initialUnits"]) < 0)

        open_orders = account["pendingOrderCount"]
        open_trades = account["openTradeCount"]
        fxidx = [p["instrument"] for p in account["positions"]].index(trader.instrument)
        pnl = float(account["positions"][fxidx][trader.instrument]["pl"]) + float(account["positions"][fxidx][trader.instrument]["unrealizedPL"])



        
        firstrow["accountID"] = trader.platform.accountID
        firstrow["name"] = trader.name
        firstrow["riskmanager"] = manager.name
        firstrow["ts"] = signal["timestamp"][-1]
        firstrow["d_int"] = trader.intakeGranularity
        firstrow["d_hor"] = trader.horizonGranularity
        firstrow["instrument"] = trader.instrument
        firstrow["regime"] = REGIME
        firstrow["B"] = signal["B"][-1]
        firstrow["S"] = signal["S"][-1]
        firstrow["U"] = signal["theta"][-1]
        firstrow["V"] = signal["mu"][-1]
        firstrow["UxV"] = "theta x mu" #dynamically pull if more fluid else hardcode
        firstrow["ActionHat"] = signal["ActionHat"][-1]
        firstrow["lotsize_roll"] = lotsize_roll
        firstrow["B_a"] = None #B_a with dynamic filling later, FOK
        firstrow["S_a"] = None #S_a with dynamic filling later, FOK
        firstrow["T_a"] = None #T_a with dynamic filling later, FOK
        firstrow["why_unfill"] = None #why_unfill with dynamic filling later, FOK
        firstrow["liquidity"] = liquidity
        firstrow["bid"] = bid
        firstrow["ask"] = ask
        firstrow["spread"] = ask - bid
        firstrow["intended"] = [ask if signal["ActionHat"][-1] == 0 else (bid if signal["ActionHat"][-1] == 1 else 0)][0]
        firstrow["fill_roll"] = fill_roll
        firstrow["slip_r_roll"] = fill_roll - firstrow["intended"]
        firstrow["slip_a_roll"] = np.abs(firstrow["slip_r_roll"])
        firstrow["slip_pips_roll"] = firstrow["slip_a_roll"] / pip_size(firstrow["instrument"])
        firstrow["open"] = signal["open"][-1] # ohlcv needs to be in signal, tune the t 2
        firstrow["high"] = signal["high"][-1]
        firstrow["low"] = signal["low"][-1] 
        firstrow["close"] = signal["close"][-1]
        firstrow["volume"] = signal["volume"][-1]
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

        return finalized_record