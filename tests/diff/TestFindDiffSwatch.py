import time
from pathlib import Path
from random import shuffle
from typing import List
from unittest import TestCase

from cv2 import cv2

import config
from diff import find_diff_random
from diff.DiffSink import DiffSink
from diff.FaceFinder import FaceFinder
from diff.FaceSquare import FaceSquare
from pipelines import first_pipeline
from services import batch_data_loader_service, video_service, image_service, file_service, face_recog_service
from services.RedisService import RedisService
from util.BatchData import BatchData

logger = config.create_logger(__name__)


FaceFinder = FaceFinder


class TestFindDiffSwatch(TestCase):

  def test_load_metadata(self):
    # Arrange
    files_c = file_service.walk_to_path(Path(config.TRAIN_PARENT_PATH_C), filename_endswith="metadata.json")
    files_d = file_service.walk_to_path(Path(config.TRAIN_PARENT_PATH_D), filename_endswith="metadata.json")
    files = files_c + files_d

    for f in files:
      batch_data: BatchData = batch_data_loader_service.load_batch_from_path(f)
      break

    # Act
    assert(len(files) > 0)

    # Assert

  def test_spot_rnd_diff(self):
    first_pipeline.pipeline_stage_3()

  def test_load_face_data(self):
    pass
    # Assert

  def test_load_face_finder(self):
    # Arrange
    filename = 'ahhdprhoww.mp4'
    vid_path = Path(f'D:\\Kaggle Downloads\\deepfake-detection-challenge\\train\\dfdc_train_part_0\\{filename}')
    key = FaceFinder.get_redis_key(vid_path)
    redis_service = RedisService()

    # Act
    face_finder: FaceFinder = redis_service.read_binary(key)

    # Assert
    assert (face_finder is not None)
    assert (len(face_finder.frame_detections.keys()) > 0)

    frame_keys = list(face_finder.frame_detections.keys())
    logger.info(f'Key: {frame_keys}')
    faces = face_finder.frame_detections[frame_keys[0]]
    assert (type(faces) == list)
    f: FaceSquare = faces[0]
    assert (type(f) == FaceSquare)
    assert (f.xmax > 0)

    logger.info(f'face_square: {f}')

  def test_load_speed(self):
    # Arrange
    logger.info("trying to connect to redis.")
    redis_service = RedisService()
    logger.info('Connected to redis!')
    key = "foo"
    # face_finder = FaceFinder()
    face_finder = {'xmin': 123, 'xmax': 234, 'ymin': 456, 'ymax': 4567}

    time_start = time.time()
    # Act
    # redis_service.write_binary(key, face_finder)
    # face_finder_actual = redis_service.read_binary(key)
    redis_service.write_as_json(key, face_finder)
    face_finder_actual = redis_service.read_as_json(key)

    time_end = time.time()

    # Assert
    logger.info(f'Time: {time_end - time_start}')
    assert (type(face_finder_actual) == FaceFinder)

  def test_get_video_frame_face(self):
    # Arrange
    filename = 'dnrpknwija.mp4'
    files = file_service.walk_to_path(Path(config.TRAIN_PARENT_PATH_D), filename_endswith=filename)
    assert(len(files) == 1)

    vid_path = files[0]
    logger.info(f'vid: {vid_path}')

    assert(vid_path.exists())

    image, _, _ = video_service.get_single_image_from_vid(vid_path, 0)

    # l1: 408,706; r1: 652:950 - swatch
    # l2: 397,812; r2: 560,976 - face
    red = (255, 0, 0)
    green = (0, 255, 0)

    l1 = (408,706)
    r1 = (652,950)

    l2 = (397,812)
    r2 = (560,976)

    image_rect_1 = cv2.rectangle(image, pt1=l1, pt2=r1, color=red, thickness=3)
    image_rect_1 = cv2.rectangle(image, pt1=l2, pt2=r2, color=green, thickness=3)

    image_service.show_image(image_rect_1, 'Original')

  def test_get_swatch_pair(self):
    # Arrange
    vid_path = Path('D:\\Kaggle Downloads\\deepfake-detection-challenge\\train\\dfdc_train_part_3\\dnrpknwija.mp4')
    image, _, _ = video_service.get_single_image_from_vid(vid_path, 0)
    image_service.show_image(image, 'Original Fake')
    redis_service = RedisService()
    red = (255, 0, 0)
    green = (0, 255, 0)

    l2 = (397, 812)
    r2 = (560, 976)

    # 397,812; 560,976

    face_finder = FaceFinder.load(redis_service, vid_path)
    # Act
    for i in range(25):
      swatch_fake, swatch_real, x_rnd, y_rnd = find_diff_random.get_swatch_pair(face_finder, 0, image, image)

      l1 = (x_rnd, y_rnd)
      r1 = (x_rnd + 244, y_rnd + 244)
      image_corner = cv2.rectangle(image, pt1=l1, pt2=r1, color=red, thickness=3)
      image_corner = cv2.rectangle(image_corner, pt1=l2, pt2=r2, color=green, thickness=3)
      image_service.show_image(image_corner, 'Original Fake with swatch')

      # image_service.show_image(swatch_fake, f'Swatch: {x_rnd}:{y_rnd}, {x_rnd + 244}:{y_rnd + 244}')

    # Assert
