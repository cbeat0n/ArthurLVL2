### transform_signal.py is a modular and agnostic framework that serves
### as the core engine of Arthur when it comes to calculations. It is a part 
### of proprietary ETL in ETL-Base.

# External Dependencies
import numpy as np
import pandas as pd
from datetime import datetime


## PARAMETERS
## TUNE FIRST FOR ABSOLUTE OPTIMIZATION
## Naturally occuring phenomena inside the model.

HALFLIFE = 5 # ewm halflife not tested yet
Z_WINDOW = 20 # z window not tested yet
RSI_PERIOD = 14 # rsi period standard is 14 , not tested yet
BETA_SMOOTHER = 0.25 # beta in {0.15,0.25,0.35,0.55,0.65,0.75}, other values not tested
EPS_BETA = 0.1 # eps_beta in {0.025,0.05,0.1,0.15} , other values not tested
BETA_B = BETA_SMOOTHER*(1+EPS_BETA) #Beta Buy 
BETA_S = BETA_SMOOTHER*(1-EPS_BETA) #Beta Sell
GS_REG = 1 # gsreg in {0,0.15,0.25,0.55,0.75,1} , other values not tested
MREG = -1 # mreg in {-1,1} , other values not tested
IREG = 1 # ireg in {-1,1} , other values not tested & mreg ireg not both -1
ECO_ROLL = 20 # ecoroll not tested yet
SMALL_LAG = 2 # small lag in {3,4,5,6,7,8,9,10,11,12,13,14,15} , other values not tested , s < l only
LARGE_LAG = 4 # large lag in {3,4,5,6,7,8,9,10,11,12,13,14,15} , other values not tested

## Observed Values of Theta and Mu as phenomena
## U and V are not going to be perfectly sortable as unsupervised cluster space.
## THETA MIN = ...
## THETA MAX = ...
## MU MIN = ...
## MU MAX = ...

## Regime is the market structure over time.
REGIME = {"ewmhalflife" : HALFLIFE, "zwindow" : Z_WINDOW, "rsi_p" : RSI_PERIOD, "beta" : BETA_SMOOTHER,
           "e_beta" : EPS_BETA, "beta_b" : BETA_B, "beta_s" : BETA_S, "gs" :  GS_REG, 
           "mreg" : MREG, "ireg" : IREG, "L_s" : SMALL_LAG, "L_l" : LARGE_LAG}


## HYPERPARAMETERS
## TUNE SECOND FOR ABSOLUTE OPTIMIZATION
## These are parameters that the computational engine needs to function
## in experimentation.

EPS_SMOOTHER = 1e-9 # smoother small , around 1e-9
MIN_PERIODS = 10 # rolling z min periods  , not tested yet , was max(10 , Z_WINDOW//5)
RFIGS = 4 # figures used for rounding

def decisionrule(sample : pd.Series):
    """
    decisionrule takes a horizon sample and decides whether to BUY , SELL , or HOLD. decisions are 
    made based on U_t and V_t which for MVP are Theta and Mu. Theta and Mu are calculated at zone 
    frequency.
    It returns an element of {0,1,2}.

    Inputs:
    sample: pd.Series - the sample row being decided on
    """

    if True:
        return 2 ## Naive Hold
    
    ## 0 is BUY
    ## 1 is SELL
    ## 2 is HOLD

    ## Parameterize UxV ( in this case Theta x Mu ) with polar, or something else.
    ## Make a nonlinear decision rule here.
        

def convert_timestamp(ts_str: str):
    """
    convert_timestamp takes a string and parses the seconds in that string since midnight of that day.
    It returns an integer.

    Inputs:
    ts_str : str - the timestamp string
    """

    dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f000Z")
    return dt.hour * 3600 + dt.minute * 60 + dt.second


def transform_to_signal(filename: str):
    """
    transform_to_signal is the financial mathematical engine for Arthur. Using proprietary technology
    an output signal is generated.
    It returns a Pandas DataFrame.

    Inputs:
    filename : str - the file location of the cleaned csv
    """
    
    proto_signal = pd.read_csv(filename)

    proto_signal["value"] = proto_signal["close"]

    ## X and Y as Jointly Sufficient Causal Oscillators become B and S. 
    proto_signal["dclose"] = proto_signal['close'].diff()
    proto_signal["sign"] = np.where(proto_signal["dclose"] > 0, 1.0, np.where(proto_signal["dclose"] < 0, -1.0, 0.0))

    ## OBV increment
    proto_signal["obv_inc"] = proto_signal["sign"] * proto_signal['volume'].fillna(0.0)

    ## Causal smoothing minimizes micronoise
    proto_signal["X_raw"] = pd.Series(proto_signal["obv_inc"], index=proto_signal.index).ewm(halflife=HALFLIFE, adjust=False).mean()

    ## Causal Rolling Z score 
    proto_signal["X_m"] = proto_signal["X_raw"].rolling(Z_WINDOW, min_periods=MIN_PERIODS).mean().shift(1)
    proto_signal["X_sd"] = proto_signal["X_raw"].rolling(Z_WINDOW, min_periods=MIN_PERIODS).std(ddof=0).shift(1)
    proto_signal["X"] = (proto_signal["X_raw"] - proto_signal["X_m"]) / (proto_signal["X_sd"].replace(0, EPS_SMOOTHER))
    
    ## X in N(0,1) distribution
    proto_signal["X"] = proto_signal["X"].fillna(EPS_SMOOTHER)
    

    ## RSI with Wilder Smoothing
    proto_signal["r"] = proto_signal['close'].diff()

    proto_signal["r_gain"] = proto_signal["r"].clip(lower=0.0)
    proto_signal["r_loss"] = (-proto_signal["r"]).clip(lower=0.0)

    ##  Wilder's smoothing
    proto_signal["avggain"] = proto_signal["r_gain"].ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    proto_signal["avgloss"] = proto_signal["r_loss"].ewm(alpha=1/RSI_PERIOD, adjust=False).mean()

    proto_signal["rs"] = proto_signal["avggain"] / proto_signal["avgloss"].replace(0.0,EPS_SMOOTHER) # Smoother
    proto_signal["rsi"] = 100 - (100 / (1 + proto_signal["rs"]))
    proto_signal["rsi"] = proto_signal["rsi"].fillna(50.0)

    ## Centered to Y in [-1,1]
    proto_signal["Y"] = (proto_signal["rsi"] - 50.0) / 50.0



    ## GS Orthonormalization with a regulator. Usually 1 gives best results.
    ## Complete orthonormalization results in X and Y different.
    if GS_REG > 0:
        Xv = proto_signal["X"].to_numpy()
        Yv = proto_signal["Y"].to_numpy()

        ## Step 1: Normalize X
        norm_X = np.linalg.norm(Xv)
        if norm_X == 0:
            norm_X = 1.0
        X_orth = Xv / norm_X

        ## Step 2: Subtract projection of Y on X to make it orthogonal
        proj = np.dot(Yv, X_orth) * X_orth
        Y_orth_raw = Yv - GS_REG*proj

        ## Step 3: Normalize Y
        norm_Y = np.linalg.norm(Y_orth_raw)
        if norm_Y == 0:
            norm_Y = 1.0
        Y_orth = Y_orth_raw / norm_Y

        ##  Replace in DataFrame
        proto_signal["X"] = X_orth
        proto_signal["Y"] = Y_orth

   
    ##  zij are weights for buying and selling on i, X and Y on j.
    ## zij > 0 for Buy and zij < 0 for Sell
    proto_signal["z11"] = proto_signal["X"]
    proto_signal["z12"] = proto_signal["Y"]
    proto_signal["z21"] = -proto_signal["X"]
    proto_signal["z22"] = -proto_signal["Y"]

    ## wij are softmax probabilities for a stochastic matrix / isomorphism
    ## from (X,Y) to (B,S).
    ## det(W) = 0 iff beta_B = - beta_S which is not possible or when X = Y which
    ## doesn't happen due to GS. 
    proto_signal["w11"] = (np.exp(BETA_B*proto_signal["z11"])) / (np.exp(BETA_B*proto_signal["z11"]) + np.exp(BETA_B*proto_signal["z12"]))
    proto_signal["w12"] = (np.exp(BETA_B*proto_signal["z12"])) / (np.exp(BETA_B*proto_signal["z11"]) + np.exp(BETA_B*proto_signal["z12"]))
    proto_signal["w21"] = (np.exp(BETA_S*proto_signal["z21"])) / (np.exp(BETA_S*proto_signal["z21"]) + np.exp(BETA_S*proto_signal["z22"]))
    proto_signal["w22"] = (np.exp(BETA_S*proto_signal["z22"])) / (np.exp(BETA_S*proto_signal["z21"]) + np.exp(BETA_S*proto_signal["z22"]))


    ## <B,S> = W<X,Y> in vector space isomorphic to R^2, (0,1)^2
    proto_signal["B"] =  proto_signal["w11"] * proto_signal["X"] + proto_signal["w12"] * proto_signal["Y"]
    proto_signal["S"] =  proto_signal["w21"] * proto_signal["X"] + proto_signal["w22"] * proto_signal["Y"]


    ## Transform to $" = B - S or M and $ = B + S or I (intensity / conviction).
    ## These are up to isomorphism and not literal.
    proto_signal["M"] = proto_signal["B"] - proto_signal["S"]
    proto_signal["I"] = proto_signal["B"] + proto_signal["S"]

    ## All transformations not necessarily strict, but always preserve sign
    ## for Bryan's heuristic.
    proto_signal["M"] = MREG*(proto_signal["M"]-proto_signal["M"].rolling(ECO_ROLL).mean())/proto_signal["M"].rolling(ECO_ROLL).std()
    proto_signal["I"] = IREG*(proto_signal["I"]-proto_signal["I"].rolling(ECO_ROLL).mean())/proto_signal["I"].rolling(ECO_ROLL).std()

    proto_signal["B"] = (proto_signal["M"] + proto_signal["I"])/2
    proto_signal["S"] = (-proto_signal["M"] + proto_signal["I"])/2

    ## Values change as intermediaries but final values themselves
    ## always preserve sign.
    Bmin = proto_signal["B"].min()
    Bmax = proto_signal["B"].max()
    Smin = proto_signal["S"].min()
    Smax = proto_signal["S"].max()

    proto_signal["B"] = (1/(Bmax-Bmin))*proto_signal["B"]+(-Bmin/(Bmax-Bmin))
    proto_signal["S"] = (1/(Smax-Smin))*proto_signal["S"]+(-Smin/(Smax-Smin))

    proto_signal["B"] = proto_signal["B"].round(RFIGS)    
    proto_signal["S"] = proto_signal["S"].round(RFIGS)

    ## Theta Calculations

    ## Use simple unit time step to avoid time-scale domination
    proto_signal["dBdt"] = proto_signal["B"].diff(LARGE_LAG) / LARGE_LAG
    proto_signal["dSdt"] = proto_signal["S"].diff(LARGE_LAG) / LARGE_LAG


    ## Two 2D vectors that encode direction of change over the last few steps
    ## vB = [1, dBdt], vS = [1, dSdt]
    proto_signal["vB_x"] = proto_signal["dBdt"].apply(lambda x: 1)
    proto_signal["vB_y"] = proto_signal["dBdt"].fillna(EPS_SMOOTHER)
    proto_signal["vS_x"] = proto_signal["dSdt"].apply(lambda x: 1)
    proto_signal["vS_y"] = proto_signal["dSdt"].fillna(EPS_SMOOTHER)

    ## Argument Precursor
    proto_signal["Dot"] = proto_signal["vB_x"] * proto_signal["vS_x"] + proto_signal["vB_y"] * proto_signal["vS_y"]
    proto_signal["normB"] = (proto_signal["vB_x"]**2 + proto_signal["vB_y"]**2)**(0.5)
    proto_signal["normS"] = (proto_signal["vS_x"]**2 + proto_signal["vS_y"]**2)**(0.5)

    ## Argument for arccos
    proto_signal["Argt"] = (proto_signal["Dot"] ) / (proto_signal["normB"] * proto_signal["normS"])
    proto_signal["Argt"] = proto_signal["Argt"].clip(-1.0,1.0)

    proto_signal["Theta"]  = proto_signal["Argt"].apply(lambda x: np.degrees(np.arccos(x)))
    proto_signal["Theta"] = proto_signal["Theta"].round(RFIGS)

    ## Crystallized Margin Mu Calculations
    proto_signal["Margin"] = proto_signal["B"] - proto_signal["S"]
    proto_signal["Margin"] = proto_signal["Margin"].round(RFIGS)

    proto_signal["Intensity"] = proto_signal["B"] + proto_signal["S"]
    proto_signal["Intensity"] = proto_signal["Intensity"].round(RFIGS)

    proto_signal["Mu_s"] = proto_signal["Margin"].diff(SMALL_LAG).apply(lambda x : EPS_SMOOTHER if np.abs(x) < EPS_SMOOTHER else x) 
    proto_signal["Mu_l"] = proto_signal["Margin"].diff(LARGE_LAG).apply(lambda x : EPS_SMOOTHER if np.abs(x) < EPS_SMOOTHER else x)

    proto_signal["Mu"] = (proto_signal["Mu_s"] - proto_signal["Mu_l"] ) / proto_signal["Mu_l"]
    proto_signal["Mu"] = proto_signal["Mu"].round(RFIGS)

    ## Apply timestamp conversion
    proto_signal["T"] = proto_signal["timestamp"].apply(convert_timestamp)


    ## Apply Decision Rule
    proto_signal["ActionHat"] = proto_signal.apply(decisionrule, axis=1)

    ## Needed columns
    cols = ["timestamp","T","open","high","low","close","volume",
                         "value","X","Y","B","S","Margin","Intensity","Theta","Mu","ActionHat"]

    ## Sentinel value for easy UI use and screening for NaN.
    proto_signal[cols] = proto_signal[cols].fillna(0.00000000314)


    ## Intermediate columns designed to be healthy. Drop them, find cols healthy.
    return {"signal" : proto_signal[cols]}