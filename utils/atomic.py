### atomic.py holds the one safe way Arthur writes a CSV to disk.

## External Dependencies
import os # Library for operating system communication
import pandas as pd # Library for data manipulation


def write_csv_atomic(df : pd.DataFrame , full_path : str):
    """
    write_csv_atomic writes a DataFrame to a CSV so that the file at full_path is always either
    the old complete file or the new complete file, never a half written one. It writes to a
    temporary file next to the target and then swaps it in with os.replace, which is a single step.
    The temporary name ends in .csv.tmp, so a search for *.csv never sees it. If anything fails
    the temporary file is removed and the error is raised.
    It returns the filepath.

    Inputs:
    df : pd.DataFrame - the DataFrame to write
    full_path : str - the finished filepath
    """

    tmp_path = f"{full_path}.{os.getpid()}.csv.tmp"

    try:
        with open(tmp_path, "w", newline = "") as f:
            df.to_csv(f, index = False)
            f.flush()
            os.fsync(f.fileno()) # the bytes are on disk before the swap
        os.replace(tmp_path , full_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return full_path
