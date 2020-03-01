from pathlib import Path
from unittest import TestCase

import config
from util import blazeface_detection
from util.FaceDetectionIterator import FaceDetectionIterator

logger = config.create_logger(__name__)


class TestBlazeFaceDetection(TestCase):

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

  def test_get_all_processed_vid_paths(self):
    # Arrange
    fdi = FaceDetectionIterator(config.FACE_DET_PAR_PATH)

    # Act
    all_paths = []
    for df_det in fdi:
      logger.info(f"Cols: {df_det.columns}")
      logger.info(f'det: {df_det.iloc[0]["detections"]}')
      break
      # all_paths.extend(df_det['path'].tolist())

    logger.info(f"Number paths: {len(all_paths)}")

    # Assert


def get_processed_paths():
  fdi = FaceDetectionIterator(config.FACE_DET_PAR_PATH)

  all_paths = []
  for df_det in fdi:
    all_paths.extend(df_det['path'].tolist())

  return all_paths
