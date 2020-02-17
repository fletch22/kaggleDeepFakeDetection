from pathlib import Path
from unittest import TestCase

import config
from util.BatchData import BatchData

logger = config.create_logger(__name__)


class TestBatchData(TestCase):

  def test_load(self):
    # Arrange
    file_path = Path(config.SMALL_HEAD_OUTPUT_PATH, 'metadata.pkl')

    # Act
    batch_data = BatchData.load(file_path)

    logger.info(batch_data.df_metadata.columns)

    # Assert
    assert (batch_data.size() > 0)
