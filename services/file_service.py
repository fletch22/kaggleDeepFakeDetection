import os
from datetime import datetime
from os import walk as walker
from pathlib import Path
from typing import List
from zipfile import ZipFile

import config
from util import random_util

logger = config.create_logger(__name__)


def walk(dir):
  file_paths = []
  for (dirpath, dirnames, filenames) in walker(dir):
    for f in filenames:
      file_paths.append(os.path.join(dirpath, f))

  return file_paths


def walk_to_path(dir: Path, filename_startswith: str = None, filename_endswith: str = None) -> List[Path]:
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


def does_file_have_bytes(path: Path):
  return path.stat().st_size > 0


def delete_files(destination: Path):
  files: List[Path] = walk_to_path(destination)

  for f in files:
    f.unlink()

  return files


def get_unique_persist_filename(parent_path: Path, base_output_stem: str, extension: str, use_date: bool = False):
  uniquer = random_util.random_string_digits(6)
  if use_date:
    uniquer = str(datetime.utcnow().isoformat()).replace(':', '-').replace('.', '-')
  file_path = Path(parent_path, f'{base_output_stem}_{uniquer}.{extension}')
  if file_path.exists():
    return get_unique_persist_filename(parent_path, base_output_stem, extension)

  return file_path


def archive_paths(many_paths: List[Path], output_path_parent: Path, output_path_stem: str, output_path_extension: str):
  output_path_unique = get_unique_persist_filename(output_path_parent, output_path_stem, output_path_extension, use_date=True)
  with ZipFile(output_path_unique, 'w') as zipObj:
    for path in many_paths:
      zipObj.write(path)

  return output_path_unique


def write_text_file(output_path, contents):
  file_path = Path(output_path)
  with open(file_path, 'w') as f:
    f.write(contents)
