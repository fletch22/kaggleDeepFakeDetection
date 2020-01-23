from pathlib import Path
from unittest import TestCase

import config
from services import clustering_service, file_services


class TestClusteringService(TestCase):

  def test_embeddings(self):
    # Arrange
    face_paths = [Path(f) for f in file_services.walk(config.SMALL_HEAD_IMAGE_PATH)]

    # Act
    clustering_service.get_embeddings(face_paths)

    # Assert