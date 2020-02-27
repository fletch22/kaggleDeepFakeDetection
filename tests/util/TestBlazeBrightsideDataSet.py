from unittest import TestCase

from services import batch_data_loader_service, image_service
from util.BatchDataRow import BatchDataRow
from util.BlazeBrightSideDataSet import BlazeBrightSideDataSet


class TestBlazeBrightsideDataSet(TestCase):

  def test_load(self):
    # Arrange
    batch_data = batch_data_loader_service.load_batch(0)
    bdr: BatchDataRow = batch_data.__getitem__(0)

    # Act
    blazeBright = BlazeBrightSideDataSet(bdr.video_path)

    # Assert
    assert(blazeBright is not None)

  def test_get_subframe_images(self):
    # Arrange
    batch_data = batch_data_loader_service.load_batch(0)
    bdr: BatchDataRow = batch_data.__getitem__(0)

    blazeBright = BlazeBrightSideDataSet(bdr.video_path)

    # Act
    images = blazeBright.get_subframe_images_info(0)

    # Assert
    assert(len(images) == 3)
    image_service.show_image(images[0])
    image_service.show_image(images[1])
    image_service.show_image(images[2])