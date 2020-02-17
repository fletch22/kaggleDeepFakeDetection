import pandas as pd
from pandas import DataFrame


def set_no_truncate_head():
  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)
  pd.set_option('display.width', None)
  pd.set_option('display.max_colwidth', -1)


def head(df: DataFrame, rows:int = 10):
  set_no_truncate_head()
  return df.head(rows)