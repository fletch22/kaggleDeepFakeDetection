import pickle
from pathlib import Path
from typing import List

import pandas as pd

import config
from diff.RandomFrameDiff import RandomFrameDiff
from services import file_service
from util import random_util

logger = config.create_logger(__name__)

class DiffSink():

  def __init__(self, output_par_path: Path, max_output_size_mb: int = 1):
    self.parent_path = output_par_path
    self.max_output_size_mb = max_output_size_mb

    file_paths = file_service.walk_to_path(output_par_path, filename_endswith=".pkl")

    self.path_map = {}

    for f in file_paths:
      with open(str(f), 'rb') as fp:
        logger.info(f'About to load pickle file \'{f.name}\' with processed data ...')
        df = pickle.load(fp)
        path_set = set(df['path'].tolist())
        for p in path_set:
          frame_index_list = df[df['path'] == p]['frame_index'].tolist()
          self.path_map[Path(p).name] = frame_index_list

    self.intialize_new_dataframe()

  def intialize_new_dataframe(self):
    self.df = pd.DataFrame(columns=['filename', 'path', 'frame_index', 'x', 'y', 'height', 'width', 'swatch_path', 'score'])

    rnd_str = random_util.random_string_digits(6)
    parent_output = Path(str(self.parent_path), "data")
    if not parent_output.exists():
      parent_output.mkdir()

    self.output_path = Path(str(parent_output), f'dataframe_{rnd_str}.pkl')

  def is_max_frames_in_video_processed(self, vp: Path, max_frames_per_video: int):
    filename = vp.name
    result = False
    if filename in self.path_map.keys():
      frames: List = self.path_map[filename]
      if len(frames) >= max_frames_per_video:
        result = True

    return result

  def is_frame_processed(self, vp: Path, frame_index):
    filename = vp.name
    result = False
    if filename in self.path_map.keys():
      frame_map = self.path_map[filename]
      if str(frame_index) in frame_map:
        result = True

    return result

  def append(self, vp: Path, f: RandomFrameDiff, swatch_path: Path):
    self.df = self.df.append({'filename': vp.name, 'path': str(vp), 'frame_index': f.frame_index, 'x': f.x, 'y': f.y, 'height': f.height, 'width': f.width, 'swatch_path': str(swatch_path), 'score': f.score},
                             ignore_index=True)

  def persist(self):
    self.df.to_pickle(self.output_path)

    # Check size:
    size = self.output_path.stat().st_size / 1000000
    if size > self.max_output_size_mb:
      self.intialize_new_dataframe()

  @staticmethod
  def get_persisted(pickle_parent_path: Path) -> pd.DataFrame:
    pickles = file_service.walk_to_path(pickle_parent_path, filename_endswith='.pkl')

    df_all = []
    for p in pickles:
      df = pd.read_pickle(str(p))
      df_all.append(df)

    return pd.concat(df_all)



