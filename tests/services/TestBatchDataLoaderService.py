from unittest import TestCase

import config
from services import batch_data_loader_service

logger = config.create_logger(__name__)


class TestBatchDataLoaderService(TestCase):

  def test_batch_load(self):
    # Arrange
    # Act
    batch_data = batch_data_loader_service.load_batch(0)

    logger.info(batch_data.df_metadata.head(20))

    # Assert
    assert (batch_data.__len__() > 0)


