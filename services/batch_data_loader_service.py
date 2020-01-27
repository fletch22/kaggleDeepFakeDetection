import pandas as pd

import config
from BatchData import BatchData, COL_CANDIDATE, COL_ORIGINAL
from services import file_service
from pathlib import Path

logger = config.create_logger(__name__)

def load_batch(index: int) -> BatchData:
  metadata_file_path = file_service.get_metadata_path_from_batch(index)
  parent_dir_path: Path = Path(metadata_file_path).parent

  df_metadata = pd.read_json(metadata_file_path)

  columns = df_metadata.columns

  df_metadata = df_metadata.T
  df_metadata[COL_CANDIDATE] = columns
  df_metadata = df_metadata.rename(columns={'original': COL_ORIGINAL})

  logger.info(df_metadata.head(10))

  data_files = file_service.walk(config.get_train_batch_path(index))
  return BatchData(df_metadata, parent_dir_path, data_files)
