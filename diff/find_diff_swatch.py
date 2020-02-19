from pathlib import Path

import imutils
import skimage
from cv2 import cv2
from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity

import config
from services import video_service, image_service
from util.BatchData import BatchData

logger = config.create_logger(__name__)


def get_diffs(batch_data: BatchData, max_diffs: int = 1):
  fakes_list = BatchData.convert(batch_data.get_fakes())

  diffs = []
  for ndx, batch_row_data in enumerate(fakes_list):
    if len(diffs) >= max_diffs:
      break

    logger.info(f"vid_path: {batch_row_data.video_path}")

    original_filename = batch_row_data.original_filename
    vid_path_fake: Path = batch_row_data.video_path

    vid_path_real = batch_data.get_vid_path(original_filename)

    fake_frame_infos = video_service.process_all_video_frames(vid_path_fake, max_process=max_diffs)
    real_frame_infos = video_service.process_all_video_frames(vid_path_real, max_process=max_diffs)

    vid = {
      batch_row_data.filename: []
    }
    diffs.append(vid)

    for image_fake_info in fake_frame_infos:
      image_fake, _, _, ndx, _ = image_fake_info

      image_real_info = real_frame_infos[ndx]
      image_real, _, _, _, _ = image_real_info

      o_height, o_width, _ = image_real.shape
      f_height, f_width, _ = image_fake.shape

      image_service.show_image(image_real, "Original")

      if f_height * f_width > o_height * o_width:
        image_real = cv2.resize(image_real, (f_width, f_height), interpolation=cv2.INTER_NEAREST)
      elif o_height * o_width > f_height * f_width:
        image_fake = cv2.resize(image_fake, (o_width, o_height), interpolation=cv2.INTER_NEAREST)

      image_rectangle_diffs = get_contour_ssim(image_fake, image_real)
      vid[batch_row_data.filename].append({
        'frame_index': ndx,
        'rect_diffs': image_rectangle_diffs
      })

      # image_real = tf.convert_to_tensor(image_real)
      # image_fake = tf.convert_to_tensor(image_fake)
      # ssim_results: tf.Tensor = ssim(image_real, image_fake, max_val=255)
      # vid[batch_row_data.filename].append({
      #   'frame_index': ndx,
      #   'ssim': ssim_results
      # })

  # sess = tf.Session()
  # with sess.as_default():
  #   for vid in diffs:
  #     for key in vid.keys():
  #       ssim_arr = vid[key]
  #       for ssim_dict in ssim_arr:
  #         ssim_dict['ssimEval'] = ssim_dict['ssim'].eval()
  #         logger.info(f"ssim_dict: {ssim_dict['ssimEval']}")

  return diffs


def get_contour_ssim(image_fake, image_real):
  gray_real = cv2.cvtColor(image_real, cv2.COLOR_BGR2GRAY)
  gray_fake = cv2.cvtColor(image_fake, cv2.COLOR_BGR2GRAY)

  (score, diff_image) = structural_similarity(gray_real, gray_fake, full=True)

  diff_image = (diff_image * 255).astype("uint8")

  print("SSIM: {}".format(score))

  # threshold the difference image, followed by finding contours to
  # obtain the regions of the two input images that differ
  thresh = 0
  max_val = 220

  # https://www.pyimagesearch.com/2014/09/29/finding-brightest-spot-image-using-python-opencv/
  # raise Exception("Add Guassian blur to get better results?")

  diff_image_blur = cv2.GaussianBlur(diff_image, (19, 19), 0)
  (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(diff_image_blur)
  cv2.circle(image_real, minLoc, 5, (255, 0, 0), 100)

  image_service.show_image(image_real, "Original")
  # image_service.show_image(image_fake, "Modified")
  image_service.show_image(diff_image, "Diff")

  # image_real, image_fake = threshholding(diff_image, image_real, image_fake, max_val, thresh)



def threshholding(diff_image, image_real, image_fake, max_val, thresh):
  blur = cv2.GaussianBlur(diff_image, (5, 5), 0)
  ret3, thresh = cv2.threshold(blur, thresh, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  # thresh = cv2.adaptiveThreshold(diff_image, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
  # thresh = cv2.threshold(diff_image, thresh, max_val, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = imutils.grab_contours(contours)
  rectangle_diffs = []
  # loop over the contours
  for c in contours:
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    tuple_rect = cv2.boundingRect(c)
    rectangle_diffs.append(tuple_rect)
    (x, y, w, h) = tuple_rect
    cv2.rectangle(image_real, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # cv2.rectangle(image_fake, (x, y), (x + w, y + h), (0, 0, 255), 2)

  image_service.show_image(image_real, "Original")
  image_service.show_image(image_fake, "Modified")
  image_service.show_image(diff_image, "Diff")
  image_service.show_image(thresh, "Thresh")

  return image_real, image_fake
