from unittest import TestCase

import config
from services import spark_service

logger = config.create_logger(__name__)


class TestSparkService(TestCase):

  def test_connection(self):
    # Arrange
    collection = ['abc', 'cde']

    # Act
    spark_service.execute(collection, process_in_spark)

    # Assert


def process_in_spark(item):
  logger.info(item)
