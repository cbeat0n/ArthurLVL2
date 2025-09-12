### extract_auto.py is a modular and agnostic framework that extracts
### the signal matrix from ETL-Base for ETL-Auto.

# Dependencies
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
import os

# Extract the signal matrix into ETL-Auto.
# Parquet later instead of .csv for efficiency but Bryan needs to understand why.


def extract_signal_matrix(signal_file):

    signal = pd.read_csv(signal_file)

    return signal


def extract_final_matrix(base_dir , instrument):

    date_str = datetime.now().strftime("%Y-%m-%d")

    fpath = Path(base_dir) / "final" / f"finalized_data_{instrument}_{date_str}.csv"

    if not fpath.exists():

        return {
            "exists" : False,
            "file" : fpath,
            "df" : np.nan
        }
    
    final_matrix = pd.read_csv(fpath)
    os.makedirs(fpath, exist_ok=True)

    return {
        "exists" : True,
        "file" : fpath,
        "df": final_matrix
    }