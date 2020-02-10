import os
from pathlib import Path
from unittest import TestCase

import numpy as np
import torch
from cv2 import cv2
from torch.utils.data.dataloader import DataLoader

import config
from services import image_service, video_service
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

  def test_blazeface_real_image(self):
    # Arrange
    # img, height, width = image_service.pick_image(2, 3, 1)
    blazeface = BlazeFace()

    batch_path = config.get_train_batch_path(0)
    # vid_filename = "ambabjrwbt.mp4"
    vid_filename = "aimzesksew.mp4"
    vid_path = Path(batch_path, vid_filename)

    logger.info(f"Odim: {video_service.get_single_image_from_vid(vid_path, 13)[0].shape}")

    blaze_dataset = BlazeDataSet(vid_path=vid_path, max_process=10)

    for i in range(1, 3):
      img = blaze_dataset.__getitem__(i)
      image_service.show_image(img)

      # logger.info(f"img.shape: {img.shape}")
      height, width, _ = img.shape
      # return

      # offset = ((height - width) // 2)
      # img = img[:width, :]

      # logger.info(f"img.shape: {img.shape}")
      # image_service.show_image(img)
      image_resize = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)

      # Act
      detections = blazeface.predict_on_image(image_resize)
      logger.info(detections)

      face_obj_list = blaze_dataset.get_face_images_in_subframe(detections, i)

      for f in face_obj_list:
        face_image = f['image']
        xmin = f['xmin']
        ymin = f['ymin']
        logger.info(f"xmind: {xmin}; {ymin}")

        image_service.show_image(face_image)

  def test_dataset_and_loader(self):
    # Arrange
    # blaze_dataset = BlazeDataSet(0, 8)
    batch_path = config.get_train_batch_path(0)
    vid_path = Path(batch_path, "ambabjrwbt.mp4")
    blaze_dataset = BlazeDataSet(vid_path=vid_path, max_process=1)

    blaze_dataloader = DataLoader(blaze_dataset, batch_size=60, shuffle=False, num_workers=0)

    blazeface = BlazeFace()

    # Act
    all_video_detections = blazeface_detection.batch_detect(blaze_dataloader, blazeface)

    blaze_dataset.merge_sub_frame_detections(all_video_detections)

    # blazeface_detection.save_cropped_blazeface_image(all_video_detections, blaze_dataset)



