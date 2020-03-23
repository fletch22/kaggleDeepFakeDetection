import shutil
from pathlib import Path

import pandas as pd

import config

logger = config.create_logger(__name__)

real_filename_full = 'df_r.pkl'
fake_filename_full = 'df_r.pkl'

real_filename_tiny = 'df_r_tiny.pkl'
fake_filename_tiny = 'df_r_tiny.pkl'


def merge_real_and_fake(destination: Path, max_process=None):
  raise Exception("Not implemented yet. Need to read existing DF to determine already processed and need to update same instead of creating new DF every time this function is run.")

  df_f, df_r = get_dataframes(max_process)

  all_fpaths, all_labels = get_labels_and_paths(df_f, df_r)

  path_images = Path(destination, 'images')
  path_images.mkdir(exist_ok=True)

  logger.info(f"Got {len(all_fpaths)} paths.")

  map = {}
  for ndx, path_str in enumerate(all_fpaths):
    label = all_labels[ndx]
    map = add_to_map(label, map, path_images, path_str)

    if ndx > 0 and ndx / 1000 == 0:
      logger.info(f'Mapped {ndx} files.')

  logger.info(f'Got map keys: {len(map.keys())} keys.')

  data_list = [map[i] for i in map.keys()]

  df = pd.DataFrame(data=data_list, columns=['path', 'score', 'original_path'])
  to_pickle(destination, df)

  assert(df.shape[0] == len(data_list))

  return map


def get_labels_and_paths(df_f, df_r):
  all_labels = []
  all_fpaths = []
  for df in list((df_f, df_r)):
    all_labels.extend(df['score'].values)
    all_fpaths.extend(df['swatch_path'].values)

  return all_fpaths, all_labels


def get_dataframes(max_process):
  df_f, df_r = getTempDataframes()

  logger.info(f'Orig df size: {df_f.shape[0]}')

  total_process = max_process // 2

  if max_process is not None:
    df_f = df_f.iloc[:total_process]
    df_r = df_r.iloc[:total_process]

  assert (df_r.shape[0] > 0)
  assert (df_f.shape[0] > 0)

  return df_f, df_r


def add_to_map(label: str, map: dict, path_images: Path, path_str: str):
  path = Path(path_str)
  parent_folder_name = path.parent.name
  new_path = Path(path_images, f'{parent_folder_name}_{path.name}')

  if not new_path.exists():
    shutil.copy(path_str, new_path)

    if str(new_path) in map.keys():
      raise Exception("Path already exists in map.")

    map[str(new_path)] = {
      'path': str(new_path),
      'score': label,
      'original_path': path_str
    }

  return map


def to_pickle(destination, df):
  data_path = Path(destination, 'data')
  data_path.mkdir(exist_ok=True)
  pickle_path = Path(data_path, 'data.pkl')

  df.to_pickle(pickle_path)


def getTempDataframes():
  filename = real_filename_full
  # filename = real_filename_tiny
  tmp_output_path_r = Path(config.TEMP_OUTPUT_PAR_PATH, filename)
  df_r = pd.read_pickle(tmp_output_path_r)

  # df_r: pd.DataFrame = find_real_swatches.load_history(max_pickles=1)
  # df_r = df_r[:1000]
  # df_r.to_pickle(str(tmp_output_path_r))

  filename = fake_filename_full
  # filename = fake_filename_tiny
  tmp_output_path_f = Path(config.TEMP_OUTPUT_PAR_PATH, filename)
  df_f = pd.read_pickle(tmp_output_path_f)

  # df_f: pd.DataFrame = DiffSink.load_history(max_pickles=1)
  # df_f = df_f[:1000]
  # df_f.to_pickle(str(tmp_output_path_f))

  return df_f, df_r
