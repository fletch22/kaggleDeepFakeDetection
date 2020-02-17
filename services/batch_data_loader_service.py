from pathlib import Path

import pandas as pd

import config
from util.BatchData import BatchData, COL_CANDIDATE, COL_ORIGINAL, COL_VID_PATH
from services import file_service

logger = config.create_logger(__name__)


def load_batch(index: int) -> BatchData:
  metadata_file_path = file_service.get_metadata_path_from_batch(index)

  df_metadata = pd.read_json(metadata_file_path)
  columns = df_metadata.columns

  df_metadata = df_metadata.T
  df_metadata[COL_CANDIDATE] = columns
  df_metadata = df_metadata.rename(columns={'original': COL_ORIGINAL})

  filePathDict = file_service.get_files_as_dict(Path(config.TRAIN_PARENT_PATH_D), ".mp4")

  def add_path(row):
    result = None
    filename = row[COL_CANDIDATE]
    if filename in filePathDict.keys():
      result = filePathDict[filename]
    return result

  df_metadata[COL_VID_PATH] = df_metadata.apply(add_path, axis=1)

  df_metadata = df_metadata[~df_metadata[COL_VID_PATH].isnull()]

  return BatchData(df_metadata)
