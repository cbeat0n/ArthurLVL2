### transform_signal.py is an an agnostic framework that serves to intake 
### cleaned data from a database that stores data after load_cleaned.py
### and turn it into a signal matrix.
### On this module B(t) , S(t), Theta(t), and ActionHat(t) will be calculated.

# Dependencies
import os
import numpy as np
import pandas as pd
from datetime import datetime


def acquire_proto_signal(filename):
    
    proto_signal_df = pd.read_csv(filename)

    return proto_signal_df


# Decision Rule Logic separate from transform to signal, to be used after.
# Consider polar or nonlinear instead of box regions in (T,M) space.
THETA_DEG = 5 # Degrees of theta angle
EPSILON_MU = 0.1 # Return of mu

def decisionrule(row):
    #intake is intake, horizon is the aggregate of intake.
    # entire signal is the horizon, and zone is the 
    # aggregate of the horizon. Make decision on horizon first then on 
    # zone. ie intake S30 horizon gran m1 zone m5 and horizon is H
    
    # Theta, Mu model dependent on regime structure in T,M space

    # Nonlinear parameterization of theta x mu to be used for decision rule
    # Later.

    if row["theta"] > THETA_DEG:
        if ( row["mu_small"] - row["mu_large"] ) / row["mu_large"] > EPSILON_MU:
            return 0 # BUY CALLSIGN
        elif ( row["mu_small"] - row["mu_large"] ) / row["mu_large"] < - EPSILON_MU:
            return 1 # SELL CALLSIGN
        else:
            return 2 # HOLD CALLSIGN DUE TO LACK OF CHANGE IN HARD/SOFT OF CURRENCY, dTheta NOISE
    else:
        return 2 # HOLD CALLSIGN 
        
# Utility: Convert timestamp to seconds since epoch
def convert_timestamp(ts_str):
    dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f000Z")
    return dt.hour * 3600 + dt.minute * 60 + dt.second


def transform_to_signal(proto_signal):

    # Parameters
    BETA_SMOOTHER = 0.25 # beta = {0.15,0.25,0.35,0.55,0.65,0.75} to start, 0.25 most fluid best separation
    EPS_BETA = 0.1 # Perturbs Beta, consider {0.025,0.05,0.1,0.15}, 0.05 best
    BETA_B = BETA_SMOOTHER*(1+EPS_BETA) #Beta Buy 
    BETA_S = BETA_SMOOTHER*(1-EPS_BETA) #Beta Sell
    GS_REG = 1 # Gram Schmidt Regularizer for partial orthonormalization try {0,0.15,0.25,0.55,0.75}
    MREG = -1 # Regulates M z score, scale and flip valid
    IREG = 1 # Regulates I z score, scale and flip valid
    SMALL_LAG = 2
    LARGE_LAG = 4
    EPS_SMOOTHER = 1e-9 # Smooth only just a little, otherwise negligible
    SLOPE_LAG = 2 # Small lag for slope calculation
    
    REGIME = (BETA_SMOOTHER , EPS_BETA, BETA_B , BETA_S , GS_REG , MREG , IREG,
              SMALL_LAG , LARGE_LAG , EPS_SMOOTHER , SLOPE_LAG , THETA_DEG , EPSILON_MU)
    
    # The Regime is the market structure over time. These are the parameters. Anything below
    # this line is a hyperparameter.

    proto_signal["value"] = proto_signal["close"]

    ## X and Y as Jointly Sufficient Causal Oscillators become B and S. 
    # X is volume based, Y is price based.
    # --- X: OBV increment (stationary) ---
    # Sign of bar close change (0 if flat)
    proto_signal["dclose"] = proto_signal['close'].diff()
    proto_signal["sign"] = np.where(proto_signal["dclose"] > 0, 1.0, np.where(proto_signal["dclose"] < 0, -1.0, 0.0))

    # OBV increment = signed volume per bar
    proto_signal["obv_inc"] = proto_signal["sign"] * proto_signal['volume'].fillna(0.0)

    # Light causal smoothing to reduce microstructure noise
    proto_signal["X_raw"] = pd.Series(proto_signal["obv_inc"], index=proto_signal.index).ewm(halflife=5, adjust=False).mean()

    # Causal rolling z-score (compute mean/std on t-1)
    z_window = 20
    proto_signal["X_m"] = proto_signal["X_raw"].rolling(z_window, min_periods=max(10, z_window//5)).mean().shift(1)
    proto_signal["X_sd"] = proto_signal["X_raw"].rolling(z_window, min_periods=max(10, z_window//5)).std(ddof=0).shift(1)
    proto_signal["X"] = (proto_signal["X_raw"] - proto_signal["X_m"]) / (proto_signal["X_sd"].replace(0, EPS_SMOOTHER))
    proto_signal["X"] = proto_signal["X"].fillna(0.0)  # safe init

    # --- Y: RSI (Wilder), centered to be stationary ---
    proto_signal["r"] = proto_signal['close'].diff()

    proto_signal["r_gain"] = proto_signal["r"].clip(lower=0.0)
    proto_signal["r_loss"] = (-proto_signal["r"]).clip(lower=0.0)

    # Wilder's smoothing is causal (EMA with alpha=1/period)
    rsi_period = 14
    proto_signal["avggain"] = proto_signal["r_gain"].ewm(alpha=1/rsi_period, adjust=False).mean()
    proto_signal["avgloss"] = proto_signal["r_loss"].ewm(alpha=1/rsi_period, adjust=False).mean()

    proto_signal["rs"] = proto_signal["avggain"] / proto_signal["avgloss"].replace(0.0, np.nan)
    proto_signal["rsi"] = 100 - (100 / (1 + proto_signal["rs"]))
    proto_signal["rsi"] = proto_signal["rsi"].fillna(50.0)  # neutral when undefined early

    # Center to ~[-1, 1]: stationary price-pressure oscillator
    proto_signal["Y"] = (proto_signal["rsi"] - 50.0) / 50.0



    #Global orthonormalization at granularity to separate signals into "Volume" and "Price"
    #based pressures
    # --- Orthonormalize X and Y using Gram–Schmidt ---
    if GS_REG > 0:
        Xv = proto_signal["X"].to_numpy()
        Yv = proto_signal["Y"].to_numpy()

        # Step 1: Normalize X
        norm_X = np.linalg.norm(Xv)
        if norm_X == 0:
            norm_X = 1.0
        X_orth = Xv / norm_X

        # Step 2: Subtract projection of Y on X to make it orthogonal
        proj = np.dot(Yv, X_orth) * X_orth
        Y_orth_raw = Yv - GS_REG*proj

        # Step 3: Normalize Y
        norm_Y = np.linalg.norm(Y_orth_raw)
        if norm_Y == 0:
            norm_Y = 1.0
        Y_orth = Y_orth_raw / norm_Y

        # Replace in DataFrame
        proto_signal["X"] = X_orth
        proto_signal["Y"] = Y_orth

   
    ##  zij are weights for buying and selling on i, X and Y on j.
    # zij > 0 for Buy and zij < 0 for Sell
    proto_signal["z11"] = proto_signal["X"]
    proto_signal["z12"] = proto_signal["Y"]
    proto_signal["z21"] = -proto_signal["X"]
    proto_signal["z22"] = -proto_signal["Y"]

    ## wij are softmax probabilities for a stochastic matrix / isomorphism lite
    ## from (X,Y) to (B,S).
    proto_signal["w11"] = (np.exp(BETA_B*proto_signal["z11"])) / (np.exp(BETA_B*proto_signal["z11"]) + np.exp(BETA_B*proto_signal["z12"]))
    proto_signal["w12"] = (np.exp(BETA_B*proto_signal["z12"])) / (np.exp(BETA_B*proto_signal["z11"]) + np.exp(BETA_B*proto_signal["z12"]))
    proto_signal["w21"] = (np.exp(BETA_S*proto_signal["z21"])) / (np.exp(BETA_S*proto_signal["z21"]) + np.exp(BETA_S*proto_signal["z22"]))
    proto_signal["w22"] = (np.exp(BETA_S*proto_signal["z22"])) / (np.exp(BETA_S*proto_signal["z21"]) + np.exp(BETA_S*proto_signal["z22"]))


    # <B,S> = W<X,Y> in vector space isomorphic to R^2.
    proto_signal["B"] =  proto_signal["w11"] * proto_signal["X"] + proto_signal["w12"] * proto_signal["Y"]
    proto_signal["S"] =  proto_signal["w21"] * proto_signal["X"] + proto_signal["w22"] * proto_signal["Y"]


    #Transform to $" = B - S or M and $ = B + S or I (intensity / conviction)
    proto_signal["M"] = proto_signal["B"] - proto_signal["S"]
    proto_signal["I"] = proto_signal["B"] + proto_signal["S"]

    # Linear scaling is monotone / order preserving isomorphism.
    # sign(M) preserved so long as Bryan and I are comfortable knowing that
    # M' != M but it isn't about maths it's about what the maths means. Margin of local regime instead of
    # Margin in total, so bot can focus on what's happening in a reasonable timeframe.
    # Therefore lookahead value is Bryan heuristic but formalized. Keeps the adding going.
    # Use mean and stddev to turn M and I into the distribution D(0,1) for whatever
    # distribution it is.
    # Rolling 20 rows not too large of % of dataset, good heuristic.
    # Scaling z score by coefficient MREG and IREG allow for scaling along M and I,
    # possibly flipping definition, e.g M' = -M means Selling is Buying and viceversa
    # where maybe increased volume is not bullish but preparing for a large short.
    proto_signal["M"] = MREG*(proto_signal["M"]-proto_signal["M"].rolling(20).mean())/proto_signal["M"].rolling(20).std()
    proto_signal["I"] = IREG*(proto_signal["I"]-proto_signal["I"].rolling(20).mean())/proto_signal["I"].rolling(20).std()

    proto_signal["B"] = (proto_signal["M"] + proto_signal["I"])/2
    proto_signal["S"] = (-proto_signal["M"] + proto_signal["I"])/2

    #Affine for (m_i,m_j) using T = aQ+b 
    #where a = 1/(max-min) and b = -min/(max-min)
    # B and S have different T but logic is same.
    # It is not a field preserving monotonic isomorphism, so BXS not preserved
    # HOWEVER new B and S preserve sign(B-S) and therefore valid
    # because Bryan's heuristic will be preserved.
    Bmin = proto_signal["B"].min()
    Bmax = proto_signal["B"].max()
    Smin = proto_signal["S"].min()
    Smax = proto_signal["S"].max()

    proto_signal["B"] = (1/(Bmax-Bmin))*proto_signal["B"]+(-Bmin/(Bmax-Bmin))
    proto_signal["S"] = (1/(Smax-Smin))*proto_signal["S"]+(-Smin/(Smax-Smin))

    proto_signal["B"] = proto_signal["B"].round(4)    
    proto_signal["S"] = proto_signal["S"].round(4)

    ### REWRITE CRYSTALLIZED MARGIN
    # Crystallized Margin coordinate -> mu_t = $"'(c)*h ~ backwards difference

    proto_signal["Margin"] = proto_signal["B"] - proto_signal["S"]

    proto_signal["Margin"] = proto_signal["Margin"].round(4)

    proto_signal["mu_small"] = proto_signal["margin"].diff(SMALL_LAG).apply(lambda x : EPS_SMOOTHER if np.abs(x) < EPS_SMOOTHER else x) 
    # Approximate backwards diff
    # with a small lag

    proto_signal["mu_large"] = proto_signal["margin"].diff(LARGE_LAG).apply(lambda x : EPS_SMOOTHER if np.abs(x) < EPS_SMOOTHER else x)
    # Approximate backwards diff
    # with a large lag

    proto_signal["mu"] = (proto_signal["mu_small"] - proto_signal["mu_large"] ) / proto_signal["mu_large"]
    proto_signal["mu"] = proto_signal["mu"].round(4)

    # Apply timestamp conversion
    proto_signal["t_sec"] = proto_signal["timestamp"].apply(convert_timestamp)


    # Use simple unit time step to avoid time-scale domination
    proto_signal["dBdt"] = proto_signal["B"].diff(SLOPE_LAG) / SLOPE_LAG
    proto_signal["dSdt"] = proto_signal["S"].diff(SLOPE_LAG) / SLOPE_LAG


    # Two 2D vectors that encode direction of change over the last K steps
    # vB = [1, dBdt], vS = [1, dSdt]
    proto_signal["vB_x"] = proto_signal["dBdt"].apply(lambda x: 1)
    proto_signal["vB_y"] = proto_signal["dBdt"].fillna(EPS_SMOOTHER)
    proto_signal["vS_x"] = proto_signal["dSdt"].apply(lambda x: 1)
    proto_signal["vS_y"] = proto_signal["dSdt"].fillna(EPS_SMOOTHER)

    # Angle between vB and vS (in degrees), robust to magnitude/offset of time
    proto_signal["dot"] = proto_signal["vB_x"] * proto_signal["vS_x"] + proto_signal["vB_y"] * proto_signal["vS_y"]
    proto_signal["normB"] = (proto_signal["vB_x"]**2 + proto_signal["vB_y"]**2)**(0.5)
    proto_signal["normS"] = (proto_signal["vS_x"]**2 + proto_signal["vS_y"]**2)**(0.5)

    # Argument for arccos
    proto_signal["argt"] = (proto_signal["dot"] ) / (proto_signal["normB"] * proto_signal["normS"])
    proto_signal["argt"] = proto_signal["argt"].clip(-1.0,1.0)

    proto_signal["theta"]  = proto_signal["argt"].apply(lambda x: np.degrees(np.arccos(x)))
    proto_signal["theta"] = proto_signal["theta"].round(4)

    
    # Calculate ActionHat
    proto_signal["ActionHat"] = proto_signal.apply(decisionrule, axis=1)

    # Sentinel values to be smoothed and replaced for historical freezing.
    proto_signal[["B", "S","margin","mu", "theta"]] = proto_signal[["B", "S", "margin","mu" ,"theta"]].fillna(-0.0314)


    # Drop intermediate columns if not needed
    return proto_signal[["timestamp","t_sec","open","high","low","close","volume",
                         "value","X","Y","B","S","margin","theta","mu","ActionHat"]]