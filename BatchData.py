from pathlib import Path
from typing import Any, Union, Optional, List

from pandas import DataFrame
from pandas.io.json._json import JsonReader

import config

logger = config.create_logger(__name__)

COL_CANDIDATE = "candidate_filename"

class BatchData():
  df_metadata: Union[Optional[JsonReader], Any] = None
  data_files: List = None
  image_dir_path: Path = None

  def __init__(self, df_metadata: DataFrame, image_dir_path: Path, data_files: List):
    self.df_metadata = df_metadata
    self.data_files = data_files
    self.image_dir_path = image_dir_path

  def size(self):
    return self.df_metadata.shape[0]

  def get_candidate_file_path(self, index: int) -> Path:
    return Path(f"{self.image_dir_path}/{self.df_metadata.iloc[index][COL_CANDIDATE]}")