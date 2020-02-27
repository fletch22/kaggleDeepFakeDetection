import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pandas import DataFrame

from services import file_service
from util import random_util
from util.FaceDetectionIterator import FaceDetectionIterator

COL_FILENAME = "filename"
COL_PATH = "path"
COL_DETECTIONS = "detections"


class FaceDetection():

  def __init__(self, output_parent_path: Path, max_mb_output_file: int = 1):
    self.output_parent_path = output_parent_path
    self.max_mb_output_file = max_mb_output_file

    self.initialize()

  def initialize(self):
    self.output_path = self.get_new_output_path()
    self.df: DataFrame = pd.DataFrame(columns=[COL_FILENAME, COL_PATH, COL_DETECTIONS])

  def add_row(self, path: Path, detections: List[List[Dict]]):
    self.df = self.df.append({COL_FILENAME: path.name, COL_PATH: str(path), COL_DETECTIONS: detections}, ignore_index=True)

  def persist(self):
    output_path_str = str(self.output_path)
    self.df.to_pickle(output_path_str)

    if not self.is_valid_path(self.output_path):
      self.initialize()

  def get_new_output_path(self):
    output_path = Path(self.output_parent_path, f"face_detection_{random_util.random_string_digits(6)}.pkl")
    if not self.is_valid_path(output_path):
      output_path = self.get_new_output_path()

    return output_path

  def is_valid_path(self, output_path: Path):
    is_valid = True
    if output_path.exists():
      size = os.path.getsize(str(output_path)) / 1000000
      if size > self.max_mb_output_file:
        is_valid = False
    return is_valid

  def get_unfiltered_video_paths(parent_folder_paths: List[str]):
    video_paths = []
    for par_path in parent_folder_paths:
      files = file_service.walk_to_path(par_path, filename_endswith=".mp4")
      video_paths.extend(files)

    return video_paths

  @staticmethod
  def get_video_paths(parent_folder_paths: List[str], output_parent_path: Path):
    video_paths = FaceDetection.get_unfiltered_video_paths(parent_folder_paths)

    df_iterator: FaceDetectionIterator = FaceDetection.get_dataframes_iterator(output_parent_path=output_parent_path)
    proc_vid_path_strs = []
    for df in df_iterator:
      proc_vid_path_strs.extend(df[COL_PATH].tolist())

    return [vp for vp in video_paths if str(vp) not in proc_vid_path_strs]

  @staticmethod
  def get_dataframes_iterator(output_parent_path: Path) -> FaceDetectionIterator:
    return FaceDetectionIterator(output_parent_path=output_parent_path)

