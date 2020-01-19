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

      video_service.process_all_video_frames(vid_path, get_face_data)
    stopwatch.stop()

    logger.info(stopwatch)


def get_face_data(image, frame_index: int, file_path: Path):
  face_infos = face_recog_service.get_face_infos(image)
  face_data = []
  all_faces = []
  for fi in face_infos:
    face_image, _, _, _, _ = fi
    face_landmarks_list = face_recognition.face_landmarks(face_image)
    all_faces.append(face_landmarks_list)
    # image_service.show_image(face_image)

  if len(all_faces) == 0:
    logger.info(f"No face found for frame {frame_index} in '{file_path.name}'.")

  face_data.append((all_faces, frame_index, file_path))

  return face_data
