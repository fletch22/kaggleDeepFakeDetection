from pathlib import Path

import pandas as pd
from fastai.vision import ImageDataBunch, List

import config

logger = config.create_logger(__name__)


# https://docs.fast.ai/vision.data.html#ImageDataBunch.from_csv

def get_data_from_path(path_image: Path):
  # tfms = get_transforms(do_flip=False)

  data = (ImageDataBunch.from_folder(path_image)
          .random_split_by_pct()
          .label_from_func(get_float_labels)
          .transform())
  # data.normalize(imagenet_stats)

  return data


def get_data_from_list(dfs: List[pd.DataFrame]):
  all_labels = []
  all_fpaths = []
  for df in dfs:
    all_labels.extend(df['score'].values)
    all_fpaths.extend(df['swatch_path'].values)

  # ImageDataBunch.from_lists()

  # data = (ImageDataBunch.from_df(df)
  #         .random_split_by_pct()
  #         .label_from_func(get_float_labels)
  #         .transform())
  # return data


def get_float_labels(path: Path):
  logger.info(f'{type(path)}')
