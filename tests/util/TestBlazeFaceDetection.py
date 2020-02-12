from unittest import TestCase

from util import blazeface_detection


class TestBlazeFaceDetection(TestCase):

  def test_multi_batch_detection(self):
    # Arrange
    # Act
    blazeface_detection.multi_batch_detection()

    # Assert