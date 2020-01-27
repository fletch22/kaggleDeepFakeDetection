from pathlib import Path
from unittest import TestCase

from BatchData import BatchData
from services import video_service, batch_data_loader_service
from services.image_service import show_image


class TestVideoServices(TestCase):

  def test_show_frame(self):
    # Arrange
    batch_data: BatchData = batch_data_loader_service.load_batch(8)
    vid_path: Path = batch_data.get_candidate_file_path(0)
    index = 100

    # Act
    image, _, _ = video_service.get_single_image_from_vid(vid_path, index)

    show_image(image, f"Image {index}: {vid_path.name}")

    # Assert
