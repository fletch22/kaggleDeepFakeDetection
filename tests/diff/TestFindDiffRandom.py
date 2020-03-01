import time
from pathlib import Path
from typing import List
from unittest import TestCase

import config
from diff import find_diff_random
from diff.FaceFinder import FaceFinder
from diff.FaceSquare import FaceSquare
from services import batch_data_loader_service
from services.RedisService import RedisService

logger = config.create_logger(__name__)


class TestFindDiffSwatch(TestCase):

  def test_spot_rnd_diff(self):
    # Arrange
    output_par_path = Path(config.SSIM_RND_DIFFS_OUTPUT_PATH)
    batch_data = batch_data_loader_service.load_batch(3)

    max_process = 1000
    max_process_per_video = 300

    # Act
    diffs = find_diff_random.get_diffs(batch_data, output_par_path, max_total_process=max_process, max_process_per_video=max_process_per_video)

    # Assert

  def test_load_face_data(self):
    pass
    # Assert

  def test_load_face_finder(self):
    # Arrange
    filename = 'ahhdprhoww.mp4'
    vid_path = Path(f'D:\Kaggle Downloads\deepfake-detection-challenge\train\dfdc_train_part_0\{filename}')
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


# FIXME: fix this.
def transform_det(vid_path: Path, detections: List, redis_service: RedisService):
  for faces_in_frame in detections:
    for ndx, face in enumerate(faces_in_frame):
      frame_index = face['frame_index']

      coords = face['coords']
      face_square = FaceSquare(coords['xmax'], coords['xmin'], coords['ymax'], coords['ymin'])

      key = get_diff_key(vid_path, frame_index, ndx)
      redis_service.write_binary(frame_index, face_square)


def get_diff_key(vid_path, frame_index, ndx):
  key = f'{vid_path.name}_{frame_index}_{ndx}'
