### extract_auto.py is a modular and agnostic framework that performs
### proprietary ETL in ETL-Auto for Arthur.

## External Dependencies
import pandas as pd # Library for data manipulation
from datetime import datetime , timezone # Library for time objects
import numpy as np # Library for numerical calculations
import os # Library for operating system communications




def extract_signal_matrix(signal_file : str):
    """
    extract_signal_matrix extracts the signal matrix from its filepath.
    It returns a JSON with the signal matrix.

    Inputs:
    signal_file : str - the file of the signal matrix
    """
    
    signal = pd.read_csv(signal_file)

    return {"signal" : signal}


def extract_final_matrix(base_dir , trader):
    """
    extract_final_matrix is a neither feed forward nor feed backward extraction function
    that makes a JSON on the existence of the finalized matrix for a given trader , instrument , 
    and day. At most one finalized record is expected daily.
    It returns a JSON with information on the finalized matrix.

    Inputs:
    base_dir : str - the base directory for the file
    trader: Trader - the trader on the loop, one trader CRUDs one finalized matrix at horizon
    """
    
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    end_dir = f"final/{today}/finalized_data_{trader.name}_{trader.config.instrument}.csv"
    full_path = os.path.join(base_dir , end_dir)

    ## Make the directory if it doesn't exist
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    if not os.path.exists(full_path):

        return {
            "exists" : False,
            "file" : full_path,
            "final" : None
        }
    
    
    final_matrix = pd.read_csv(full_path)
    
    return {
        "exists" : True,
        "file" : full_path,
        "final": final_matrix
    }