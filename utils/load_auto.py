### load_auto.py is a modular and agnostic framework for ETL-Auto that loads
### the finalized matrix into the data/today/final folder.

# Dependencies
import pandas as pd # Current L process is .csv locally
import os
from datetime import datetime



def load_final_data(final_data,finalfilepath):

    final_data.to_csv(finalfilepath, index=False)

    return finalfilepath