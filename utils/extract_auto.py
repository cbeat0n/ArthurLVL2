### extract_auto.py is a modular and agnostic framework that extracts
### the signal matrix from ETL-Base for ETL-Auto.

# Dependencies
import pandas as pd
from pathlib import Path
from datetime import datetime , timezone
import numpy as np
import os
from utils.operators import Trader

# Extract the signal matrix into ETL-Auto.
# Parquet later instead of .csv for efficiency but Bryan needs to understand why.


def extract_signal_matrix(signal_file):

    signal = pd.read_csv(signal_file)

    return signal


def extract_final_matrix(base_dir , trader : Trader):

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    os.makedirs(Path(base_dir) / "final", exist_ok=True)
    fpath =  Path(base_dir) /  "final" / f"finalized_data_{trader.name}_{trader.instrument}_{date_str}.csv"

    if not fpath.exists():

        return {
            "exists" : False,
            "file" : fpath,
            "df" : None
        }
    
    final_matrix = pd.read_csv(fpath)
    

    return {
        "exists" : True,
        "file" : fpath,
        "df": final_matrix
    }