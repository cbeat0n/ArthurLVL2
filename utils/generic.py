### generic.py is a modular and agnostic framework for blank slate information
### that neutralizes circular imports by being a neutral parent module.

## External Dependencies
import pandas as pd # Library for data manipulation
from typing import Protocol, Optional, Any, Iterable, Dict ## Library for generic Objects

## Generic methods and classes are left blank for iterability and to prevent circular imports.
from utils.extract_quantitative import fetch_candles
from utils.transform_cleaned import convert_to_tabular , clean_tabular
from utils.load_cleaned import  load_cleaned_data
from utils.transform_signal import  decisionrule , transform_to_signal
from utils.load_signal import load_signal_matrix

from utils.extract_auto import extract_signal_matrix , extract_final_matrix
from utils.transform_auto import finalize_records
from utils.load_auto import load_final_data
