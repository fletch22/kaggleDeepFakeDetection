import os
import shutil
from pathlib import Path
from unittest import TestCase

from cv2 import cv2

import config
from services import file_service, batch_data_loader_service

logger = config.create_logger(__name__)


class FileServicesTest(TestCase):

  def test_get_train_batch_path(self):
    # Arrange
    # Act
    dir_path = config.get_train_batch_path(0)

    # Assert
    dir_path is not None
    os.path.exists(dir_path)

  def test_get_metadata_path(self):
    # Arrange
    # Act
    metadata_path = file_service.get_metadata_path_from_batch(0)

    # Assert
    os.path.exists(metadata_path)

  def test_get_video_info(self):
    # Arrange
    bad_data = batch_data_loader_service.load_batch(0)

    path_file = bad_data.get_candidate_file_path(1)

    logger.info(path_file)

    v_cap = cv2.VideoCapture(str(path_file))
    # v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # logger.info(f"Num frames: {v_len}")

    # Act
    # Assert

  def test_archive_paths(self):
    # Arrange
    tmp_path = Path(config.TEMP_OUTPUT_PAR_PATH, 'archive_paths')
    if tmp_path.exists():
      shutil.rmtree(tmp_path)
    assert(not tmp_path.exists())
    tmp_path.mkdir(exist_ok=True)

    file_path_1 = Path(tmp_path, "test_1.txt")
    file_path_2 = Path(tmp_path, "test_2.txt")
    file_path_3 = Path(tmp_path, "test_3.txt")

    file_service.write_text_file(file_path_1, 'test1')
    file_service.write_text_file(file_path_2, 'test2')
    file_service.write_text_file(file_path_3, 'test3')

    output_path = Path(tmp_path, 'test.zip')
    # Act
    output_path = file_service.archive_paths([file_path_1, file_path_2, file_path_3], tmp_path, "archive", "zip")

    logger.info(f'{output_path}')


    # Assert
    assert(output_path.exists())
