### load_cleaned.py is an an agnostic framework that serves to load cleaned data
### from the ETLTL pipeline into a storage framework that remains agnostic.
### Currently this file is small because Pandas is very high level but any time 
### S3 or Redshift or PostgreSQL etc are introduced this will get more complicated.




# Dependencies
import pandas as pd # Current L process is .csv locally
import os
from datetime import datetime


# Function creates a folder for today if not already existing
def get_today_folder(base_dir):
    # Format date as YYYY-MM-DD
    today = datetime.now().strftime("%Y-%m-%d")
    full_path = os.path.join(base_dir, today)

    # Make the directory if it doesn't exist
    os.makedirs(full_path, exist_ok=True)
    return full_path

# Load the cleaned data into today's folder as cleaned_data_instrument_timestamp.

def load_cleaned_data(filepath , cleaned_data , instrument , safe_timestamp):

    filename = f"{filepath}/cleaned_data_{instrument}_{safe_timestamp}.csv"
    cleaned_data.to_csv(filename, index=False)

    return filename
