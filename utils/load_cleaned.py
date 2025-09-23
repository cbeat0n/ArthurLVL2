### load_cleaned.py is a modular and agnostic framework that performs proprietary 
### ETL as a part of ETL-Base for Arthur.

## Internal Dependencies
import pandas as pd # Library for data manipulation
import os # Library for operating system communication
from datetime import datetime , timezone # Library for time objects


def load_cleaned_data(base_dir : str , cleaned_data : pd.DataFrame , instrument : str, safe_timestamp : str):
    """
    load_cleaned_data intakes the Pandas DataFrame from transform_cleaned.py and sends it to proprietary
    storage, currently local as .csv.
    It returns a JSON containing the filename.

    Inputs:
    filepath : str - the filepath to have information sent to
    cleaned_data : pd.DataFrame - the DataFrame containing the cleaned data.
    instrument : str - the financial instrument being used
    safe_timestamp : str - the timestamp that designates the most recent sample
    """

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    end_dir = f"cleaned/{today}/cleaned_data_{instrument}_{safe_timestamp}.csv"
    full_path = os.path.join(base_dir,end_dir)

    ## Make the directory if it doesn't exist
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    cleaned_data.to_csv(full_path, index=False)

    return {"file" : full_path}
