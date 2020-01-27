import os
from os import walk as walker

import config


def walk(dir):
  file_paths = []
  for (dirpath, dirnames, filenames) in walker(dir):
    for f in filenames:
      file_paths.append(os.path.join(dirpath, f))

  return file_paths

def get_metadata_path_from_batch(index: int):
  batch_path = config.get_train_batch_path(index)
  return os.path.join(batch_path, config.METADATA_FILENAME)