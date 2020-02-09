import os
from pathlib import Path
from unittest import TestCase

import numpy as np
import torch
from cv2 import cv2
from torch.utils.data.dataloader import DataLoader

import config
from services import image_service
from util import blazeface_detection
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

  def test_blace_real_image(self):
    # Arrange
    # img, height, width = image_service.pick_image(2, 3, 1)

    batch_path = config.get_train_batch_path(0)
    vid_path = Path(batch_path, "ambabjrwbt.mp4")
    blaze_dataset = BlazeDataSet(vid_path=vid_path, max_process=10)
    img = blaze_dataset.__getitem__(0)

    blazeface = BlazeFace()

    image_service.show_image(img)
    logger.info(f"img.shape: {img.shape}")
    height, width, _ = img.shape

    image_service.show_image(img)

    center_offset = ((width - height) // 2)
    img = img[:, center_offset:height + center_offset]
    logger.info(f"img.shape: {img.shape}")
    image_service.show_image(img)
    return
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
    # blaze_dataset = BlazeDataSet(0, 8)
    batch_path = config.get_train_batch_path(0)
    vid_path = Path(batch_path, "ambabjrwbt.mp4")
    blaze_dataset = BlazeDataSet(vid_path=vid_path)

    assert (len(blaze_dataset.originals) == len(blaze_dataset.image_infos))

    blaze_dataloader = DataLoader(blaze_dataset, batch_size=60, shuffle=False, num_workers=0)

    blazeface = BlazeFace()

    # Act
    all_video_detections = blazeface_detection.batch_detect(blaze_dataloader, blazeface)

    blazeface_detection.save_cropped_blazeface_image(all_video_detections, blaze_dataset)



