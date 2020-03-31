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


def get_metadata_json_files(which_drive: str):
  if which_drive.lower() == 'c':
    return file_service.walk_to_path(Path(config.TRAIN_PARENT_PATH_C), filename_endswith="metadata.json")
  else:
    return file_service.walk_to_path(Path(config.TRAIN_PARENT_PATH_D), filename_endswith="metadata.json")


def get_all_metadata(which_drive: str):
  logger.info("About to get all metadata JSON files ...")
  files = get_metadata_json_files(which_drive)
  df_all = [load_batch_from_path(f).df_metadata for f in files]

  logger.info('About to concatenate all metadata JSON files.')
  return pd.concat(df_all)


# NOTE: 2020-03-06: So far data only from drive D has been processed.
def save_all_metadata_to_single_df():
  raise Exception("Disabled 'save_all_metadata_to_single_df' for now to prevent overwriting D drive data.")
  # df = get_all_metadata()
  # df.to_pickle(config.DF_ALL_METADATA_PATH)


# NOTE: 2020-03-06: This is only from drive D.
def read_all_metadata_as_df() -> pd.DataFrame:
  logger.info("About to read all metadata as a dataframe ...")
  return pd.read_pickle(config.DF_ALL_METADATA_PATH)
