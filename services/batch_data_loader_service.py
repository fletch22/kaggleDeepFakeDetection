from pathlib import Path

import pandas as pd

import config
from services import file_service
from util.BatchData import BatchData, COL_CANDIDATE, COL_ORIGINAL, COL_VID_PATH

logger = config.create_logger(__name__)


def load_batch(index: int) -> BatchData:
  metadata_file_path = file_service.get_metadata_path_from_batch(index)

  return load_batch_from_path(Path(metadata_file_path))


def load_batch_from_path(path: Path) -> BatchData:
  par_path = path.parent
  logger.info(f'About to get batch metadata.json data from \'{path.parent.name}\'')
  df_metadata = pd.read_json(path)
  columns = df_metadata.columns

  df_metadata = df_metadata.T
  df_metadata[COL_CANDIDATE] = columns
  df_metadata = df_metadata.rename(columns={'original': COL_ORIGINAL})

  filePathDict = file_service.get_files_as_dict(par_path, ".mp4")

  def add_path(row):
    result = None
    filename = row[COL_CANDIDATE]
    if filename in filePathDict.keys():
      result = filePathDict[filename]
    return result

  df_metadata[COL_VID_PATH] = df_metadata.apply(add_path, axis=1)

  df_metadata = df_metadata[~df_metadata[COL_VID_PATH].isnull()]

  logger.info("Loaded metadata.json data.")

  return BatchData(df_metadata)
