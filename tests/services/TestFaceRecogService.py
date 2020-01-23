import os
from pathlib import Path
from unittest import TestCase

import face_recognition
from stopwatch import Stopwatch

import config
from BatchData import BatchData
from services import face_recog_service, batch_data_loader_service, video_service, image_service

logger = config.create_logger(__name__)


class TestFaceRecogService(TestCase):

  def test_get_face(self):
    # Arrange
    # Batch 8, video 0 has fake
    batch_data: BatchData = batch_data_loader_service.load_batch(0)
    vid_path = batch_data.get_candidate_file_path(0)

    # Act
    image = video_service.get_single_image_from_vid(vid_path, 150)

    image_service.show_image(image)

    # Act
    face_infos = face_recog_service.get_face_infos(image)

    for fi in face_infos:
      title = f"Found {len(face_infos)} face(s)."
      face_image, _, _, _, _ = fi
      image_service.show_image(face_image, title)
      face_lines_image = face_recog_service.add_face_lines(face_image)
      image_service.show_image(face_lines_image)

    # Assert
    assert (len(face_infos) > 0)

  def test_get_face_rate(self, ):
    # Arrange
    batch_data: BatchData = batch_data_loader_service.load_batch(0)

    logger.info("Got batch data.")

    stopwatch = Stopwatch()
    stopwatch.start()
    num_videos = 1  # batch_data.size()
    for i in range(num_videos):
      logger.info(f"Getting {i}th video.")
      vid_path = batch_data.get_candidate_file_path(i)

      video_service.process_all_video_frames(vid_path, face_recog_service.get_face_data)
    stopwatch.stop()

    logger.info(stopwatch)

  def test_get_face_rate_with_spark(self, ):
    # Arrange
    batch_data: BatchData = batch_data_loader_service.load_batch(0)

    logger.info("Got batch data.")

    stopwatch = Stopwatch()
    stopwatch.start()
    num_videos = 1  # batch_data.size()
    all_face_data = []
    for i in range(num_videos):
      logger.info(f"Getting {i}th video.")
      vid_path = batch_data.get_candidate_file_path(i)
      face_data = video_service.process_all_video_frames_with_spark(vid_path)
      all_face_data.append(face_data)

    stopwatch.stop()

    logger.info(stopwatch)

  def test_haar_cascade(self):
    # Arrange
    image_path: Path = Path(config.WOMAN_PROFILE_IMAGE_PATH)
    # Act
    face_recog_service.get_face_data_from_profile(image_path)
    # Assert


