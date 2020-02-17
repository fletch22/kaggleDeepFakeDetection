from pathlib import Path
from unittest import TestCase

import pandas as pd

import config
from services import file_service
from util import blazeface_detection
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



