from unittest import TestCase

from BatchData import BatchData
from services import face_recog_service, batch_data_loader_service, video_service, image_service


class TestFaceRecogService(TestCase):

  def test_get_face(self):
    # Arrange
    # Batch 8, video 0 has fake
    batch_data: BatchData = batch_data_loader_service.load_batch(8)
    vid_path = batch_data.get_candidate_file_path(12)

    # Act
    image = video_service.get_image_from_vid(vid_path, 150)

    # Act
    face_infos = face_recog_service.get_face_infos(image)

    for fi in face_infos:
      title = f"Found {len(face_infos)} face(s)."
      face_image, _, _, _, _ = fi
      image_service.show_image(face_image, title)
      face_lines_image = face_recog_service.add_face_lines(face_image)
      image_service.show_image(face_lines_image)

    # Assert
    assert(len(face_infos) > 0)