import os
from os import walk as walker
from pathlib import Path

import config

logger = config.create_logger(__name__)

def walk(dir):
  file_paths = []
  for (dirpath, dirnames, filenames) in walker(dir):
    for f in filenames:
      file_paths.append(os.path.join(dirpath, f))

  return file_paths


def walk_to_path(dir, filename_startswith: str = None, filename_endswith: str = None):
  file_paths = walk(str(dir))
  paths = [Path(f) for f in file_paths]

  if filename_endswith is not None:
    paths = [p for p in paths if str(p.name).endswith(filename_endswith)]
  if filename_startswith is not None:
    paths = [p for p in paths if str(p.name).startswith(filename_startswith)]
  return paths


def get_metadata_path_from_batch(index: int):
  batch_path = config.get_train_batch_path(index)
  return os.path.join(batch_path, config.METADATA_FILENAME)


def get_files_as_dict(path: Path, endswith: str):
  files = walk(str(path))
  files_filtered = filter(lambda f: f.endswith(endswith), files)
  return {Path(f).name: Path(f) for f in files_filtered}
