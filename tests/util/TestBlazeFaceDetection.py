from pathlib import Path
from unittest import TestCase

import pandas as pd

import config
from services import file_service
from util import blazeface_detection
from util.BlazeDataSet import BlazeDataSet
from util.FaceDetection import FaceDetection, COL_PATH
from util.FaceDetectionIterator import FaceDetectionIterator

logger = config.create_logger(__name__)


class TestBlazeFaceDetection(TestCase):

  def test_multi_batch_detection(self):
    # Arrange
    output_parent_path = config.FACE_DET_PAR_PATH
    parent_folder_paths = [config.TRAIN_PARENT_PATH_C, config.TRAIN_PARENT_PATH_D]
    video_paths = FaceDetection.get_video_paths(parent_folder_paths=parent_folder_paths, output_parent_path=output_parent_path)

    # Act
    blazeface_detection.multi_batch_detection(video_paths=video_paths, output_parent_path=output_parent_path)

    # Assert
    df_iterator: FaceDetectionIterator = FaceDetection.get_dataframes_iterator(output_parent_path=output_parent_path)

    for df in df_iterator:
      logger.info(f"Number of rows from first pickle file: {df.shape[0]}")
      break

  def test_multi_batch_detection_2(self):
    # Arrange
    output_parent_path = config.TRASH_PATH
    parent_folder_paths = [config.TRAIN_PARENT_PATH_C]

    video_paths = [Path(config.TRAIN_PARENT_PATH_D, "dfdc_train_part_0", "aaqaifqrwn.mp4")]
    # video_paths = FaceDetection.get_video_paths(parent_folder_paths=parent_folder_paths, output_parent_path=output_parent_path)

    # Act
    dets = blazeface_detection.multi_batch_detection_2(video_paths=video_paths, output_parent_path=output_parent_path)

    # Assert
    logger.info(dets)
    # df_iterator: FaceDetectionIterator = FaceDetection.get_dataframes_iterator(output_parent_path=output_parent_path)
    #
    # for df in df_iterator:
    #   logger.info(f"Number of rows from first pickle file: {df.shape[0]}")
    #   break

  def test_get_all_processed_vid_paths(self):
    # Arrange
    fdi = FaceDetectionIterator(config.FACE_DET_PAR_PATH)

    # Act
    all_paths = []
    for df_det in fdi:
      # logger.info(f"Cols: {df_det.columns}")
      all_paths.extend(df_det['path'].tolist())

    logger.info(f"Number paths: {len(all_paths)}")

    # Assert

def get_processed_paths():
  fdi = FaceDetectionIterator(config.FACE_DET_PAR_PATH)

  all_paths = []
  for df_det in fdi:
    all_paths.extend(df_det['path'].tolist())

  return all_paths






