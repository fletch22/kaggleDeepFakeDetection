from pathlib import Path
from unittest import TestCase

import torch
from cv2 import cv2
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

import config
from BatchData import BatchData
from services import batch_data_loader_service, video_service, image_service
from util.BlazeDataSet import BlazeDataSet
from util.blazeface import BlazeFace
import numpy as np

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
    img, height, width = image_service.pick_image(2, 3, 1)

    blazeface = BlazeFace()
    blazeface.load_anchors("C:\\Users\\Chris\\workspaces\\kaggleDeepFakeDetection\\util\\anchors.npy")
    blazeface.load_weights("C:\\Users\\Chris\\workspaces\\kaggleDeepFakeDetection\\util\\blazeface.pth")

    image_service.show_image(img)
    logger.info(f"img.shape: {img.shape}")
    height, width, _ = img.shape

    center_offset = ((width - height)//2)
    img = img[:, center_offset:height + center_offset]
    logger.info(f"img.shape: {img.shape}")
    image_service.show_image(img)
    img_resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

    # return
    detections = blazeface.predict_on_image(img_resized)
    logger.info(detections.size())

    # Act
    # h = blazeface(unsqueezed)
    # h = blazeface.predict_on_batch(unsqueezed)
    # logger.info(h[0])

  def test_dataset_and_loader(self):
    # Arrange
    blaze_dataset = BlazeDataSet(2, 12)

    assert(len(blaze_dataset.originals) == len(blaze_dataset.image_infos))

    blaze_dataloader = DataLoader(blaze_dataset, batch_size=60, shuffle=False, num_workers=0)

    blazeface = BlazeFace()
    blazeface.load_anchors("C:\\Users\\Chris\\workspaces\\kaggleDeepFakeDetection\\util\\anchors.npy")
    blazeface.load_weights("C:\\Users\\Chris\\workspaces\\kaggleDeepFakeDetection\\util\\blazeface.pth")

    # Act
    for i_batch, sample_batched in enumerate(blaze_dataloader):
      h = blazeface.predict_on_batch(sample_batched.detach().numpy())
      num_faces = np.sum([item.detach().cpu().numpy().shape[0] for item in h])

      logger.info(f"Number of faces found: {num_faces}")
