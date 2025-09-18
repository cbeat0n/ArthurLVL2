### load_signal.py is an an agnostic framework that serves to load the signal matrix
### from transform_cleaned.py into a database framework, which is also agnostic.
### Currently this file is small because Pandas is very high level but any time 
### S3 or Redshift or PostgreSQL etc are introduced this will get more complicated.


# Dependencies
import pandas as pd


# Load the signal matrix into today's folder as signal_data_instrument_timestamp.

def load_signal_matrix(filepath , signal_matrix , instrument , safe_timestamp):

    signal_filename = f"{filepath}/signal_data_{instrument}_{safe_timestamp}.csv"
    signal_matrix.to_csv(signal_filename, index=False)

    return signal_filename
