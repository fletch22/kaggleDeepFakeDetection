import os
from pathlib import Path
from unittest import TestCase

from stopwatch import Stopwatch

import config
from util.BatchData import BatchData
from services import batch_data_loader_service
from services.mtcnn import face_recog_mtcnn_service

logger = config.create_logger(__name__)


class TestFaceRecogMtcnnService(TestCase):

  def test_mtcnn_face_recog(self):
    # Arrange
    image_path: Path = Path(config.WOMAN_PROFILE_IMAGE_PATH)
    # Act
    result = face_recog_mtcnn_service.get_single_face(image_path=image_path)

    # Assert
    logger.info(result)

  def test_mtcnn_get_many_face_recog_expanded(self):
    # Arrange
    batch_data: BatchData = batch_data_loader_service.load_batch(0)
    output_path = Path(config.TRASH_PATH)
    expand_frame = True
    # output_path = Path(config.TINY_IMAGE_PATH)
    # expand_frame = False

    stopwatch = Stopwatch()
    stopwatch.start()
    result = face_recog_mtcnn_service.get_faces(batch_data, output_path, expand_frame)
    stopwatch.stop()

    logger.info(f"Elapsed time: {stopwatch}")
    # Assert
    logger.info(result)

  def test_mtcn_get_tiny_faces(self):
    # Arrange
    batch_data: BatchData = batch_data_loader_service.load_batch(0)
    root_path = Path(config.SMALL_HEAD_OUTPUT_PATH)
    output_path = Path(os.path.join(config.TINY_IMAGE_PATH))

    # Act
    face_recog_mtcnn_service.find_tiny_faces(root_path, output_path)
