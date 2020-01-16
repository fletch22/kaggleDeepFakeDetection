from pathlib import Path
from typing import Any, Union, Optional, List

from pandas import DataFrame
from pandas.io.json._json import JsonReader


class BatchData():
  df_metadata: Union[Optional[JsonReader], Any] = None
  data_files: List = None
  image_dir_path: Path = None

  def __init__(self, df_metadata: DataFrame, image_dir_path: Path, data_files: List):
    self.df_metadata = df_metadata
    self.data_files = data_files
    self.image_dir_path = image_dir_path
