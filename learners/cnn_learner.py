from pathlib import Path

import numpy as np
from fastai.vision import ImageDataBunch

import config
from services import pickle_service

logger = config.create_logger(__name__)

def get_databunch():
  df = get_decorated_df()

  data = (ImageDataBunch.from_df(df, size=224)
          .random_split_by_pct()
          .transform())

  return data

def get_decorated_df(pct_train: int = 80, pct_test: int = 5):
  df, _ = pickle_service.concat_pickled_dataframes(config.MERGED_SWATCH_DATA_PATH)

  df['video_name_stem'] = df['path'].apply(lambda x: Path(x).stem.split('_')[0])
  df['gross_label'] = df['path'].apply(lambda x: 'real' if x.endswith('1.0.png') else 'fake')

  video_names = df['video_name_stem'].unique()

  np.random.shuffle(video_names)

  num_rows = video_names.shape[0]

  train_num = int((num_rows * (pct_train/100)))
  test_num = int((num_rows * (pct_test/100)))
  val_num = num_rows - train_num - test_num
  vid_train = video_names[:train_num]
  vid_validation = video_names[train_num:train_num + val_num]

  logger.info(f'Will attempt to set rows for training: {vid_train.shape[0]}.')

  vid_train_list = list(vid_train)

  def split_train_test_val(value):
    t_label = 'test'
    if value in vid_train_list:
      t_label = 'train'
    elif value in vid_validation:
      t_label = 'validation'
    return t_label

  df['test_train_split'] = df['video_name_stem'].apply(split_train_test_val)

  train_rows = df[df['test_train_split'] == 'train'].shape[0]
  val_rows = df[df['test_train_split'] == 'validation'].shape[0]
  test_rows = df[df['test_train_split'] == 'test'].shape[0]

  logger.info(f'Train {train_rows} rows.')
  logger.info(f'Validation {val_rows} rows.')
  logger.info(f'Test {test_rows} rows.')

  logger.info(f'Head: {df.head()}')