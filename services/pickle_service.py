import pandas as pd

import config
from services import file_service

logger = config.create_logger(__name__)


def get_all_dataframes(output_par_path):
  pickles = file_service.walk_to_path(output_par_path, filename_endswith='.pkl')
  pickles_filtered = [p for p in pickles if file_service.does_file_have_bytes(p)]

  logger.info(f'About to collect all {len(pickles_filtered)} pickle paths in list ...')
  all_df = [pd.read_pickle(str(p)) for p in pickles_filtered]

  return all_df


def concat_pickled_dataframes(output_par_path, max_pickles=None):
  all_df = get_all_dataframes(output_par_path)

  if max_pickles is not None and len(all_df) > max_pickles:
    all_df = all_df[:max_pickles]

  logger.info("About to concat all dataframes ...")
  df = None
  if len(all_df) > 0:
    df = pd.concat(all_df)

  return df, all_df
