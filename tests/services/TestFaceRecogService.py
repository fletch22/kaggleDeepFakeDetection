import os
from pathlib import Path
from typing import List, Dict
from unittest import TestCase

from cv2 import cv2
from pandas import DataFrame
from stopwatch import Stopwatch

import config
from BatchData import BatchData
from services import face_recog_service, batch_data_loader_service, video_service, image_service, file_service
from services.image_service import get_image_differences

logger = config.create_logger(__name__)


class TestFaceRecogService(TestCase):

  def test_get_face(self):
    # Arrange
    # Batch 8, video 0 has fake
    batch_data: BatchData = batch_data_loader_service.load_batch(0)
    vid_path = batch_data.get_candidate_file_path(0)

    # Act
    image, _, _ = video_service.get_single_image_from_vid(vid_path, 150)

    image_service.show_image(image)

    height, width, _ = image.shape

    # Act
    face_infos = face_recog_service.get_face_infos(image, height, width)

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
    max_process = 4
    batch_data: BatchData = batch_data_loader_service.load_batch(0)

    logger.info("Got batch data.")

    stopwatch = Stopwatch()
    stopwatch.start()
    num_videos = 1  # batch_data.size()
    for i in range(num_videos):
      logger.info(f"Getting {i}th video.")
      vid_path = batch_data.get_candidate_file_path(i)

      fil = video_service.process_all_video_frames(vid_path, face_recog_service.get_face_data, max_process)
    stopwatch.stop()

    for fi in fil:
      landmarks = fi['face_info_landmarks']
      frame_index = fi['frame_index']
      file_path = fi['file_path']
      for ndx, land in enumerate(landmarks):
        face_image = land['face_image']
        image_service.show_image(face_image, 'test')

        par_path = os.path.join(config.TRASH_PATH, f"{file_path.stem}")
        os.makedirs(par_path, exist_ok=True)

        image_path = os.path.join(par_path, f"{frame_index}_{ndx}.png")
        logger.info(f"Writing to {image_path}.")
        image_converted = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_path, image_converted)

    logger.info(stopwatch)

  def test_get_face_rate_with_spark(self, ):
    # Arrange
    # batch_data: BatchData = batch_data_loader_service.load_batch(0)
    batch_0_dir_path = config.get_train_batch_path(0)

    logger.info("Got batch data.")

    stopwatch = Stopwatch()
    stopwatch.start()
    num_videos = 1  # batch_data.size()
    all_face_data = []
    for i in range(num_videos):
      logger.info(f"Getting {i}th video.")
      # vid_path = batch_data.get_candidate_file_path(i)
      vid_path = Path(os.path.join(batch_0_dir_path, "tqsratgvwa.mp4"))
      # vid_path = Path("D:\\Kaggle Downloads\\deepfake-detection-challenge\\output\\trash\\owxbbpjpch")
      face_data = video_service.process_all_video_frames_with_spark(vid_path)
      all_face_data.append(face_data)

    stopwatch.stop()

    logger.info(stopwatch)

  def test_haar_cascade(self):
    # Arrange
    image_path: Path = Path(config.WOMAN_PROFILE_IMAGE_PATH)

    image = cv2.imread(str(image_path))
    height, width, _ = image.shape

    # Act
    face_recog_service.get_haar_face_data(image_path, height, width)
    # Assert

  def test_get_differences(self):
    # Arrange
    batch_data: BatchData = batch_data_loader_service.load_batch(0)

    # Act
    diffs = face_recog_service.get_face_diffs(batch_data, max_diffs=6000)
    mean = float(sum(d['ssim'] for d in diffs)) / len(diffs)

    # Assert
    logger.info(f"diffs found: {len(diffs)}; Mean: {mean}")




def find_matching_headshot(all_small_heads: List[str], fake_file_info: List[Dict], frame_index: int, head_index: int) -> Path:
  for f in fake_file_info:
    file_path = f['file_path']
    par_path = os.path.dirname(str(file_path))

    file_to_find = os.path.join(par_path, f"{frame_index}_{head_index}.png")

    logger.info(f"File_2_find: {file_to_find}")

    if file_to_find in all_small_heads:
      return Path(file_to_find)



