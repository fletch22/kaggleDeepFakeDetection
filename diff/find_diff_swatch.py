import os
from pathlib import Path

from cv2 import cv2
from numpy import float32
from pandas import DataFrame
from tensorflow.python.ops.image_ops_impl import ssim
import tensorflow as tf
import config
from services import file_service, video_service, image_service
from util.BatchData import BatchData

logger = config.create_logger(__name__)


def get_diffs(batch_data: BatchData, max_diffs: int = 1):
  fakes_list = BatchData.convert(batch_data.get_fakes())

  diffs = []
  for ndx, batch_row_data in enumerate(fakes_list):
    if len(diffs) > max_diffs:
      break

    original_filename = batch_row_data.original_filename
    vid_path_fake: Path = batch_row_data.video_path

    vid_path_real = batch_data.get_vid_path(original_filename)

    fake_frame_infos = video_service.process_all_video_frames(vid_path_fake, max_process=max_diffs)
    real_frame_infos = video_service.process_all_video_frames(vid_path_real, max_process=max_diffs)

    for image_fake_info in fake_frame_infos:
      image_fake, _, _, ndx, _ = image_fake_info

      image_real_info = real_frame_infos[ndx]
      image_real, _, _, _, _ = image_real_info

      # image_o = cv2.imread(str(original_file_path))
      # o_height, o_width, _ = image_o.shape
      #
      # image_f = cv2.imread(str(fake_file_path))
      # f_height, f_width, _ = image_f.shape
      #
      # if f_height * f_width > o_height * o_width:
      #   image_o = cv2.resize(image_o, (f_width, f_height), interpolation=cv2.INTER_NEAREST)
      # else:
      #   image_f = cv2.resize(image_f, (o_width, o_height), interpolation=cv2.INTER_NEAREST)
      #
      # image_o_conv = cv2.cvtColor(image_o, cv2.COLOR_BGR2RGB)
      # image_f_conv = cv2.cvtColor(image_f, cv2.COLOR_BGR2RGB)

      # image_real_conv = cv2.cvtColor(image_real, cv2.COLOR_BGR2RGB)
      # image_fake_conv = cv2.cvtColor(image_fake, cv2.COLOR_BGR2RGB)

      tmp_real = "D:\\Kaggle Downloads\\deepfake-detection-challenge\\output\\test_tiny_heads\\0_0.jpg"
      tmp_fake = "D:\\Kaggle Downloads\\deepfake-detection-challenge\\output\\test_tiny_heads\\31_1.jpg"

      image_real = cv2.imread(tmp_real)
      image_fake = cv2.imread(tmp_fake)

      o_height, o_width, _ = image_real.shape
      f_height, f_width, _ = image_fake.shape

      if f_height * f_width > o_height * o_width:
        image_real = cv2.resize(image_real, (f_width, f_height), interpolation=cv2.INTER_NEAREST)
      else:
        image_fake = cv2.resize(image_fake, (o_width, o_height), interpolation=cv2.INTER_NEAREST)

      image_service.show_image(image_fake)
      image_service.show_image(image_real)

      image_real = tf.convert_to_tensor(image_real)
      image_fake = tf.convert_to_tensor(image_fake)

      sess = tf.Session()
      with sess.as_default():
        ssim_results: tf.Tensor = ssim(image_real, image_fake, max_val=255)
        logger.info(ssim_results.eval())

      # if ssim_results > 0.93:
      #   image_service.show_image(image_real, "real")
      #   image_service.show_image(image_fake, "fake")
      #
      #   diffs.append(ssim_results)
      #
      #   raise Exception("Foo")

      # return dict(original_image=image_o_conv, fake_image=image_f_conv, ssim=ssim(image_o, image_f, multichannel=True))
      #   image_diffs = get_image_differences(file_path, fake_path)
      #   ssim = image_diffs['ssim']
      #   if ssim > 0.93:
      #     image_service.show_image(image_diffs['original_image'], f'original {ssim}')
      #     image_service.show_image(image_diffs['fake_image'], f'fake {ssim}')
      #   head_diff = dict(ssim=ssim, original_path=file_path, fake_path=fake_path)
      #   diffs.append(head_diff)

  return diffs