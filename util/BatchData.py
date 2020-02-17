import json
from pathlib import Path
from typing import Union, List

import pandas as pd
from pandas import DataFrame

import config
import shutil

from util import pd_util
from util.BatchDataRow import BatchDataRow

logger = config.create_logger(__name__)

COL_CANDIDATE = "candidate_filename"
COL_ORIGINAL = "original_filename"
COL_VID_PATH = "vid_path"
COL_FACE_DETECTIONS = "face_detections"
COL_FAKE_OR_REAL = "label"
COL_SPLIT = "split"

LABEL_FAKE = "FAKE"
LABEL_REAL = "REAL"

class BatchData():
  df_metadata: DataFrame = None

  def __init__(self, df_metadata: DataFrame):
    self.df_metadata: DataFrame = df_metadata

    # if COL_FACE_DETECTIONS not in self.df_metadata.columns:
    #   self.df_metadata[COL_FACE_DETECTIONS] = None

  def __len__(self):
    return self.df_metadata.shape[0]

  def __getitem__(self, ndx):
    return BatchDataRow(self.df_metadata.iloc[ndx, :])

  def get_fakes(self) -> DataFrame:
    return self.df_metadata[self.df_metadata[COL_FAKE_OR_REAL] == "FAKE"]

  @staticmethod
  def convert(df: DataFrame):
    return [BatchDataRow(df.iloc[i, :]) for i in range(df.shape[0])]

  def get_candidate_file_path(self, index: int) -> Union[Path, None]:
    return Path(self.df_metadata.iloc[index][COL_VID_PATH])

  def get_vid_path(self, candidate_filename: str):
    logger.info(candidate_filename)

    df_filtered = self.df_metadata[self.df_metadata[COL_CANDIDATE] == candidate_filename]
    if df_filtered.shape[0] == 0:
      raise Exception("Could not find candidate.")

    return Path(df_filtered.iloc[0][COL_VID_PATH])

  # def add_face_detections(self, index, detections: List):
  #   self.df_metadata.iloc[index, self.df_metadata.columns.get_loc(COL_FACE_DETECTIONS)] = self.encode_detections(detections)
  #
  # def encode_detections(self, detections):
  #   return json.dumps(detections).encode('utf-8')
  #
  # def decode_detections(self, encoded):
  #   return json.loads(encoded.decode('utf-8'))
  #
  # def persist(self, output_path: Path):
  #   self.df_metadata.to_pickle(str(output_path))
  #
  # def filter_for_face_detection(self):
  #   return self.df_metadata[self.df_metadata[COL_FACE_DETECTIONS].notnull()]
  #
  # @staticmethod
  # def load(file_path: Path):
  #   return BatchData(pd.read_pickle(str(file_path)))
