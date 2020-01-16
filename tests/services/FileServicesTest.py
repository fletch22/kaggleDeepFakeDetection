import os
from os import walk as walker
from unittest import TestCase

import config
from services import file_services


class FileServicesTest(TestCase):

  def test_get_train_batch_path(self):
    # Arrange
    # Act
    dir_path = config.get_train_batch_path(0)

    # Assert
    dir_path is not None
    os.path.exists(dir_path)

  def test_get_metadata_path(self):
    # Arrange
    # Act
    metadata_path = file_services.get_metadata_path_from_batch(0)

    # Assert
    os.path.exists(metadata_path)
