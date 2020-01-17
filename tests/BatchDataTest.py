import os
from unittest import TestCase

import config
from services import batch_data_loader_service

logger = config.create_logger(__name__)

class BatchDataTest(TestCase):

  def test_get_file(self):
    # Arrange
    batch_data = batch_data_loader_service.load_batch(0)

    # Act
    file_path = batch_data.get_candidate_file_path(0)

    logger.info(file_path)

    # Assert
    assert (os.path.exists(file_path))
