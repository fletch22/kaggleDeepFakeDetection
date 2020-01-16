import pandas as pd

import config
from BatchData import BatchData
from services import file_services


def load_batch(index: int) -> BatchData:
  metadata_file_path = file_services.get_metadata_path_from_batch(index)
  df_metadata = pd.read_json(metadata_file_path).T

  data_files = file_services.walk(config.get_train_batch_path(index))
  return BatchData(df_metadata, data_files)
