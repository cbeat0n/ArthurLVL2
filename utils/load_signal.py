### load_signal.py is a modular and agnostic framework that performs proprietary ETL
### as a part of ETL-Base for Arthur.

## Internal Dependencies
import pandas as pd # Library for data manipulation
import os # Library for operating system communication
from datetime import datetime , timezone # Library for time objects 


def load_signal_matrix(base_dir : str , signal_matrix : pd.DataFrame, instrument : str , safe_timestamp : str):
    """
    load_signal_matrix intakes the signal DataFrame from transform_signal.py and sends it to 
    its filepath.
    It returns a JSON containing the filepath.

    Inputs:
    base_dir : str - the base directory to be used
    signal_matrix : pd.DataFrame - the signal to be sent to filepath
    instrument : str - the financial instrument being traded
    safe_timestamp : str - the most recent timestamp of the data
    """

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    end_dir = f"signal/{today}/signal_data_{instrument}_{safe_timestamp}.csv"
    full_path = os.path.join(base_dir,end_dir)

    ## Make the directory if it doesn't exist
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    signal_matrix.to_csv(full_path, index=False)

    return {"file" : full_path}
