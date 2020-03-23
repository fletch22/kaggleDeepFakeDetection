from pathlib import Path

import numpy as np

import config
from pipelines.Pipeline import Pipeline
from services import pickle_service

logger = config.create_logger(__name__)


class DecorateDf(Pipeline):

  def __init__(self, root_output: Path, pipeline_dirname: str):
    Pipeline.__init__(self, root_output=root_output, pipeline_dirname=pipeline_dirname, overwrite_old_outputs=True, start_output_keys={'output_path'})

  def start(self, **kwargs):
    pct_train = kwargs['pct_train']
    pct_test = kwargs['pct_test']

    df, _ = pickle_service.concat_pickled_dataframes(config.MERGED_SWATCH_DATA_PATH)

    df['filename'] = df['path'].apply(lambda x: Path(x).name)
    df['video_name_stem'] = df['path'].apply(lambda x: Path(x).stem.split('_')[0])
    df['gross_label'] = df['path'].apply(lambda x: 'real' if x.endswith('1.0.png') else 'fake')

    video_names = df['video_name_stem'].unique()

    np.random.shuffle(video_names)

    num_rows = video_names.shape[0]

    train_num = int((num_rows * (pct_train / 100)))
    test_num = int((num_rows * (pct_test / 100)))
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

    result = {'output_path': self.persist_output_dataframe(df)}
    self.validate_start_output(result)

    return result
