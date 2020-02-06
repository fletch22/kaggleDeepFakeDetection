from pathlib import Path
from unittest import TestCase

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

import config
from BatchData import BatchData
from services import batch_data_loader_service, video_service, image_service
from util.BlazeDataSet import BlazeDataSet
from util.blazeface import BlazeFace

logger = config.create_logger(__name__)


class TestBlazeFace(TestCase):

  def testBlaze(self):
    # Arrange
    x = torch.randn(1, 3, 128, 128)
    blazeface = BlazeFace()

    # Act
    h = blazeface(x)

    # Assert
    logger.info(h)

  def testBlazeRealImage(self):
    # Arrange
    img, height, width = image_service.pick_image(8, 0, 123)

    trans = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((128, 128)),
      transforms.ToTensor()])

    img_tensor = trans(img)
    unsqueezed = img_tensor.unsqueeze(0)

    logger.info(unsqueezed.shape)

    blazeface = BlazeFace()
    h = blazeface(unsqueezed)

    logger.info(h)
    return

    blaze_dataset = BlazeDataSet(8, 12)

    img = blaze_dataset.__getitem__(4)
    image_service.show_image(img, 'test')

    blaze_dataloader = DataLoader(blaze_dataset, batch_size=24, shuffle=True, num_workers=2)

    # Act
    # h = blazeface(img_tensor)

    # Assert
