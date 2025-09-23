### load_auto.py is a modular and agnostic framework that performs
### proprietary ETL in ETL-Auto for Arthur.

## External Dependencies
import pandas as pd # Library for data manipulation
import os # Library for operating system communications
from datetime import datetime # Library for time objects



def load_final_data(final_data,final_filepath):
    """
    load_final_data loads the finalized record to storage.
    It returns a JSON containing the filepath.

    Inputs:
    final_data : pd.DataFrame - the finalized records
    final_filepath : str - the finalized records filepath
    """
    
    final_data.to_csv(final_filepath, index=False)

    return {"file" : final_filepath}