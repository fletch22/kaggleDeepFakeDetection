from pathlib import Path
from unittest import TestCase

import torch
from cv2 import cv2
from torch.utils.data.dataloader import DataLoader

import config
from services import image_service, video_service, batch_data_loader_service
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

    batch_path = config.get_train_batch_path(2)
    # vid_filename = "ambabjrwbt.mp4"
    # vid_filename = "aimzesksew.mp4"
    vid_filename = "aejroilouc.mp4"
    vid_path = Path(batch_path, vid_filename)

    logger.info(f"Odim: {video_service.get_single_image_from_vid(vid_path, 13)[0].shape}")

    blaze_dataset = BlazeDataSet(vid_path=vid_path, max_process=10)

    for i in range(1, 3):
      sub_frame_img = blaze_dataset.__getitem__(i)
      # image_service.show_image(sub_frame_img)

      # logger.info(f"img.shape: {img.shape}")
      height, width, _ = sub_frame_img.shape
      # return

      # offset = ((height - width) // 2)
      # img = img[:width, :]

      # logger.info(f"img.shape: {img.shape}")
      # image_service.show_image(img)
      sub_frame_image_resized = cv2.resize(sub_frame_img, (128, 128), interpolation=cv2.INTER_NEAREST)

      # Act
      subframe_detections = blazeface.predict_on_image(sub_frame_image_resized)
      logger.info(subframe_detections)

      face_obj_list = blaze_dataset.get_face_images_in_subframe(subframe_detections, i)

      for f in face_obj_list:
        face_image = f['image']
        xmin = f['xmin']
        ymin = f['ymin']
        logger.info(f"xmind: {xmin}; {ymin}")

        image_service.show_image(face_image)

  def test_dataset_and_loader(self):
    # Arrange

    batch_data = batch_data_loader_service.load_batch(0)
    batch_row_index = 2
    vid_path = batch_data.get_candidate_file_path(batch_row_index)

    # Act
    blaze_dataset = BlazeDataSet(vid_path, max_process=10)
    # batch_path = config.get_train_batch_path(2)
    # vid_path = Path(batch_path, "ambabjrwbt.mp4")
    # vid_path = Path(batch_path, "aimzesksew.mp4")
    # vid_path = Path(batch_path, "aejroilouc.mp4")
    # blaze_dataset = BlazeDataSet(vid_path=vid_path, max_process=1)

    if len(blaze_dataset.originals) == 0:
      raise Exception("Not enough files to process.")

    blaze_dataloader = DataLoader(blaze_dataset, batch_size=60, shuffle=False, num_workers=0)

    blazeface = BlazeFace()

    # Act
    all_video_detections = blazeface_detection.batch_detect(blaze_dataloader, blazeface)

    merged_vid_detections = blaze_dataset.merge_sub_frame_detections(all_video_detections)

    output_folder_path = Path(config.SMALL_HEAD_OUTPUT_PATH)
    output_folder_path.mkdir(exist_ok=True)

    blazeface_detection.save_cropped_blazeface_image(merged_vid_detections, blaze_dataset, output_folder_path)

    batch_data.add_face_detections(batch_row_index, merged_vid_detections)

    output_file_path = Path(output_folder_path, 'metadata.pkl')
    batch_data.persist(output_file_path)

    # batch_data_loaded = BatchData.load(output_file_path)






