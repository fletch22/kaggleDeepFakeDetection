import shutil
from pathlib import Path
from typing import List, Set

import pandas as pd

import config
from services import pickle_service, file_service

logger = config.create_logger(__name__)


class Pipeline():

  def __init__(self, root_output: Path, pipeline_dirname: str, overwrite_old_outputs: bool = False, start_output_keys: Set[str] = None):
    if start_output_keys is None:
      start_output_keys = list()
    self.root_output = root_output
    self.child_dir_output = Path(self.root_output, pipeline_dirname)
    self.child_dir_output.mkdir(exist_ok=True)

    self.dataframes_path = Path(self.child_dir_output, 'dataframes')
    self.dataobjects_path = Path(self.child_dir_output, 'dataobjects')

    self.base_output_stem = "df"
    self.start_output_keys = {} if start_output_keys is None else start_output_keys

    if overwrite_old_outputs:
      if self.dataframes_path.exists():
        shutil.rmtree(self.dataframes_path)
      if self.dataobjects_path.exists():
        shutil.rmtree(self.dataobjects_path)

    self.dataframes_path.mkdir(exist_ok=True)
    self.dataobjects_path.mkdir(exist_ok=True)

  def read_existing_output_dataframe(self) -> pd.DataFrame:
    df, _ = pickle_service.concat_pickled_dataframes(self.dataframes_path)

    return df

  def start(self, **kwargs):
    raise Exception("Not implemented yet. You need to create a method in your subclass.")

  def persist_output_dataframe(self, df: pd.DataFrame, ensure_output_name_unique: bool = False):
    output_path = Path(self.dataframes_path, f'{self.base_output_stem}.pkl')
    if ensure_output_name_unique:
      output_path = file_service.get_unique_persist_filename(self.dataframes_path, self.base_output_stem, "pkl")

    df.to_pickle(output_path)

    return output_path

  def consolidate_persisted_dataframes(self):
    logger.info(f'Output path: {self.dataframes_path}')

    df, all_df_paths = pickle_service.concat_pickled_dataframes(self.dataframes_path)

    archive_path = file_service.archive_paths(all_df_paths, self.dataframes_path, 'archive', 'pkl')
    assert (archive_path.exists())

    for f in all_df_paths:
      f.unlink()

    output_path = self.persist_output_dataframe(df)

    return output_path, archive_path

  def validate_start_output(self, start_func_result):
    keys = start_func_result.keys()
    if keys != self.start_output_keys:
      raise Exception(f'Start result dictionary keys do not match. Expected: {keys}. Actual: {self.start_output_keys}')

