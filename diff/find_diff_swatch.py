from pathlib import Path
from typing import Dict, List

import imutils
import pandas as pd
from cv2 import cv2
from skimage.metrics import structural_similarity

import config
from services import video_service, image_service
from util.BatchData import BatchData
from util.BatchDataRow import BatchDataRow
from util.list_utils import chunks

logger = config.create_logger(__name__)


def get_diffs(batch_data: BatchData, output_path: Path, max_diffs: int = None) -> List:
  fakes_list: List[BatchDataRow] = BatchData.convert(batch_data.get_fakes())

  processed_fakes = get_processed_vids(output_path)

  fakes_filterd = [f for f in fakes_list if f.filename not in processed_fakes]

  logger.info(f"unfiltered: {len(fakes_list)}")
  logger.info(f"filtered: {len(fakes_filterd)}")

  ssim_list = []
  for ndx, batch_row_data in enumerate(fakes_filterd):
    original_filename = batch_row_data.original_filename

    vid_path_fake: Path = batch_row_data.video_path
    vid_path_real = batch_data.get_vid_path(original_filename)

    ssim_data = {
      'vid_path_real': vid_path_real,
      'vid_path_fake': vid_path_fake
    }
    ssim_list.append(ssim_data)

  fakes_chunked = chunks(ssim_list, 4)
  chunked_list = list(fakes_chunked)

  all_diffs = []
  for c in chunked_list:
    # vid_diffs = spark_service.execute(c, spark_process_diff, num_slices=2)

    vid_diffs = []
    for d in c:
      diff = process_diff(d)
      vid_diffs.append(diff)

    persist(vid_diffs, output_path)
    all_diffs.extend(vid_diffs)

  return all_diffs


def process_diff(data: Dict):
  vid_path_real = data['vid_path_real']
  vid_path_fake: Path = data['vid_path_fake']

  fake_frame_infos = video_service.process_all_video_frames(vid_path_fake, max_process=None)
  real_frame_infos = video_service.process_all_video_frames(vid_path_real, max_process=None)

  diffs = get_all_vid_hotspots(fake_frame_infos, real_frame_infos)
  return {
    'vid_path_fake': vid_path_fake,
    'diffs': diffs,
  }


def persist(diffs: List[Dict], output_path: Path):
  output_path_str = str(output_path)
  if output_path.exists():
    df = pd.read_pickle(output_path_str)
  else:
    df = pd.DataFrame(columns=['fake_filename', 'path', 'diffs'])

  for d in diffs:
    vp: Path = d['vid_path_fake']
    df = df.append({'fake_filename': vp.name, 'path': str(vp), 'diffs': d['diffs']}, ignore_index=True)

  df.to_pickle(output_path_str)


def get_processed_vids(output_path: Path):
  output_path_str = str(output_path)
  df = pd.read_pickle(output_path_str)

  return df['fake_filename'].tolist()


def get_all_vid_hotspots(fake_frame_infos, real_frame_infos):
  frame_diff_hotspots = []
  for image_fake_info in fake_frame_infos:
    image_fake, _, _, ndx, _ = image_fake_info

    image_real_info = real_frame_infos[ndx]
    image_real, _, _, _, _ = image_real_info

    o_height, o_width, _ = image_real.shape
    f_height, f_width, _ = image_fake.shape

    if f_height * f_width > o_height * o_width:
      image_real = cv2.resize(image_real, (f_width, f_height), interpolation=cv2.INTER_NEAREST)
    elif o_height * o_width > f_height * f_width:
      image_fake = cv2.resize(image_fake, (o_width, o_height), interpolation=cv2.INTER_NEAREST)

    x_diff, y_diff = get_contour_ssim(image_fake, image_real)
    frame_diff_hotspots.append({
      'frame_index': ndx,
      'diff_location': {'x': x_diff, 'y': y_diff}
    })
  return frame_diff_hotspots


def get_contour_ssim(image_fake, image_real):
  gray_real = cv2.cvtColor(image_real, cv2.COLOR_BGR2GRAY)
  gray_fake = cv2.cvtColor(image_fake, cv2.COLOR_BGR2GRAY)

  (score, diff_image) = structural_similarity(gray_real, gray_fake, full=True)

  diff_image = (diff_image * 255).astype("uint8")

  print("SSIM: {}".format(score))

  # https://www.pyimagesearch.com/2014/09/29/finding-brightest-spot-image-using-python-opencv/
  diff_image_blur = cv2.GaussianBlur(diff_image, (19, 19), 0)
  (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(diff_image_blur)

  x, y = minLoc
  return x, y


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
