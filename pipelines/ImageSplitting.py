from pathlib import Path

import pandas as pd

import config
from pipelines.Pipeline import Pipeline

logger = config.create_logger(__name__)

TRAIN_SUFFIX = 'tra'
VAL_SUFFIX = 'val'
TEST_SUFFIX = 'tst'

FILE_SUFFIXES = [TRAIN_SUFFIX, VAL_SUFFIX, TEST_SUFFIX]


class ImageSplitting(Pipeline):

  def __init__(self, root_output: Path, pipeline_dirname: str, overwrite_old_outputs: bool = False):
    Pipeline.__init__(self, root_output=root_output, pipeline_dirname=pipeline_dirname, overwrite_old_outputs=overwrite_old_outputs, start_output_keys={'output_path'})

  def start(self, **kwargs):
    max_process = kwargs['max_process']
    input_path = kwargs['input_path']

    df = pd.read_pickle(input_path)

    if max_process is not None:
      df = df.iloc[:max_process, :]

    self.make_output_paths()

    new_row_list = []
    for ndx, row in df.iterrows():
      new_row = self.get_new_path(row)
      if new_row is not None:
        new_row_list.append(new_row)

    df_new = pd.DataFrame(data=new_row_list)

    logger.info(f'df_new rows: {df_new.shape[0]}')

    output_path = self.persist_output_dataframe(df=df)

    result = {'output_path': output_path}
    self.validate_start_output(result)

    return result

  def make_output_paths(self):
    train_path = Path(str(self.dataobjects_path), 'train')
    train_path.mkdir(exist_ok=True)
    val_path = Path(str(self.dataobjects_path), 'valid')
    val_path.mkdir(exist_ok=True)
    test_path = Path(str(self.dataobjects_path), 'test')
    test_path.mkdir(exist_ok=True)
    Path(train_path, '1').mkdir(exist_ok=True)
    Path(train_path, '0').mkdir(exist_ok=True)
    Path(val_path, '1').mkdir(exist_ok=True)
    Path(val_path, '0').mkdir(exist_ok=True)
    Path(test_path, '1').mkdir(exist_ok=True)
    Path(test_path, '0').mkdir(exist_ok=True)

  def has_not_been_processed(self, row):
    path = row['path']
    stem = path.stem
    return not stem.startswith('real_') and not stem.startswith('fake_')

  def get_new_path(self, row):
    path = Path(row['path'])

    if not path.exists():
      return None

    rof = row['real_or_fake_digit']
    test_train_split = row['test_train_split']
    split_suffix = TEST_SUFFIX
    if test_train_split == 'train':
      split_suffix = TRAIN_SUFFIX
    elif test_train_split == 'validation':
      split_suffix = VAL_SUFFIX

    new_path = Path(str(self.dataobjects_path), test_train_split, str(rof), f'{str(rof)}_{path.stem}_{split_suffix}.png')

    if not new_path.exists():
      path.rename(new_path)

    row['path'] = new_path
    row['filename'] = new_path.name

    return row
