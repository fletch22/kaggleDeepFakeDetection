from typing import Any, Union, Optional, List

from pandas import DataFrame
from pandas.io.json._json import JsonReader


class BatchData():
  df_metadata: Union[Optional[JsonReader], Any] = None
  data_files: List = None

  def __init__(self, df_metadata: DataFrame, data_files: List):
    self.df_metadata = df_metadata
    self.data_files = data_files
