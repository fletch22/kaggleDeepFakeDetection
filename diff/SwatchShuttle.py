from pathlib import Path
from typing import List

import pandas as pd

import config
from diff.FakeSwatchData import FakeSwatchData
from services import file_service, pickle_service
from util import random_util

logger = config.create_logger(__name__)


class SwatchShuttle():

  def __init__(self, fake_frames: List[FakeSwatchData], real_video_path: Path, output_par_path: Path):
    self.fake_frames = fake_frames
    self.real_video_path = real_video_path
    self.output_par_path = output_par_path

    self.swatch_par_path = Path(str(self.output_par_path), 'swatches')
    self.swatch_par_path.mkdir(exist_ok=True)

    self.data_dir_path = Path(str(self.output_par_path), 'data')
    self.data_dir_path.mkdir(exist_ok=True)

    self.df = pd.DataFrame(columns=['filename', 'path', 'frame_index', 'height', 'width', 'x', 'y', 'swatch_path', 'score'])

  def __len__(self):
    return len(self.fake_frames)

  def __getitem__(self, ndx):
    return self.fake_frames[ndx]

  def get_swatch_path(self, index: int):
    frame = self.fake_frames[index]
    real_filestem = self.real_video_path.stem
    dir_path = Path(self.swatch_par_path, real_filestem)
    dir_path.mkdir(exist_ok=True)
    return Path(dir_path, f'{frame.frame_index}_1.0.png')

  def save_data(self, index: int, swatch_path: Path):
    frame = self.fake_frames[index]
    self.df = self.df.append({'filename': self.real_video_path.name,
                    'frame_index': frame.frame_index,
                    'x': frame.x,
                    'y': frame.y,
                    'width': frame.width,
                    'height': frame.height,
                    'swatch_path': swatch_path,
                    'score': 1.0}, ignore_index=True)

  def persist_to_disk(self):
    file_path = self.get_unique_persist_path()
    self.df.to_pickle(file_path)

  def get_unique_persist_path(self):
    rnd = random_util.random_string_digits(6)
    file_path = Path(str(self.data_dir_path), f'swatch_shuttle_{rnd}.pkl')
    if file_path.exists():
      return self.get_unique_persist_path()
    return file_path

  @staticmethod
  def consolidate_pickles(output_path: Path):
    output_par_path = output_path.parent

    logger.info(f'Parent path: {output_par_path}')
    assert(output_par_path.exists())

    df, _ = pickle_service.concat_pickled_dataframes(output_par_path)

    logger.info("About to pickle.")
    df.to_pickle(output_path)



