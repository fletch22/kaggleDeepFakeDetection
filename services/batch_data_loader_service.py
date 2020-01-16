import pandas as pd

import config
from BatchData import BatchData
from services import file_services
from pathlib import Path

def load_batch(index: int) -> BatchData:
  metadata_file_path = file_services.get_metadata_path_from_batch(index)
  parent_dir_path: Path = Path(metadata_file_path).parent
  df_metadata = pd.read_json(metadata_file_path).T

  data_files = file_services.walk(config.get_train_batch_path(index))
  return BatchData(df_metadata, parent_dir_path, data_files)
