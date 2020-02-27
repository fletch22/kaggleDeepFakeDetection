import math
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from cv2 import cv2
from skimage.metrics import structural_similarity

import config
from services import video_service
from util import list_utils
from util.BatchData import BatchData
from util.BatchDataRow import BatchDataRow

logger = config.create_logger(__name__)


class VideoPair():
  def __init__(self, vid_path_real: Path, vid_path_fake: Path):
    self.vid_path_real = vid_path_real
    self.vid_path_fake = vid_path_fake


class RandomFrameDiff():
  def __init__(self, image: np.ndarray, frame_index: int, x: int, y: int, height: int, width: int, score: float):
    self.image = image
    self.frame_index = frame_index
    self.x = x
    self.y = y
    self.height = height
    self.width = width
    self.score = score


class RandomVidDiffs():
  def __init__(self, vid_path: Path, diffs: List[RandomFrameDiff]):
    self.vid_path = vid_path
    self.diffs = diffs


def get_diffs(batch_data: BatchData, output_path: Path, max_diffs: int = None) -> List[RandomVidDiffs]:
  fakes_filterd = get_unproc_fakes(batch_data, output_path)

  vid_pair_list = []
  for ndx, batch_row_data in enumerate(fakes_filterd):
    original_filename = batch_row_data.original_filename

    vid_path_fake: Path = batch_row_data.video_path
    vid_path_real = batch_data.get_vid_path(original_filename)

    vid_pair_list.append(VideoPair(vid_path_real, vid_path_fake))

  fakes_chunked = list_utils.chunks(vid_pair_list, 1)
  chunked_list = list(fakes_chunked)

  for c in chunked_list:
    vid_diffs = []
    for d in c:
      diff = process_diff(d)
      vid_diffs.append(diff)

    persist(vid_diffs, output_path)
    raise Exception("foo")


def get_unproc_fakes(batch_data, output_path):
  fakes_list: List[BatchDataRow] = BatchData.convert(batch_data.get_fakes())
  processed_fakes = get_processed_vids(output_path)
  fakes_filterd = [f for f in fakes_list if f.filename not in processed_fakes]

  logger.info(f"Skipping {len(fakes_list) - len(fakes_filterd)} already processed files. Will process {len(fakes_filterd)} files.")

  return fakes_filterd


def process_diff(sd: VideoPair) -> RandomVidDiffs:
  fake_frame_infos = video_service.process_all_video_frames(sd.vid_path_fake, max_process=None)
  real_frame_infos = video_service.process_all_video_frames(sd.vid_path_real, max_process=None)

  diffs = get_random_diffs(fake_frame_infos, real_frame_infos)

  return RandomVidDiffs(sd.vid_path_fake, diffs)


def persist(diffs: List[RandomVidDiffs], output_path: Path):
  output_path_str = str(output_path)
  swatches = Path(str(output_path.parent), 'swatches')
  if not swatches.exists():
    swatches.mkdir()

  if output_path.exists():
    df = pd.read_pickle(output_path_str)
  else:
    df = pd.DataFrame(columns=['filename', 'path', 'frame_index', 'x', 'y', 'height', 'width', 'swatch_path', 'score'])

  for d in diffs:
    vp: Path = d.vid_path
    image_dir_path = Path(str(swatches), vp.stem)
    if not image_dir_path.exists():
      image_dir_path.mkdir()

    for f in d.diffs:
      score_str = str(round(f.score, 5) * 100000)
      swatch_path = Path(str(image_dir_path), f'{f.frame_index}_{score_str}.png')
      cv2.imwrite(str(swatch_path), f.image)
      df = df.append({'filename': vp.name, 'path': str(vp), 'frame_index': f.frame_index, 'x': f.x, 'y': f.y, 'height': f.height, 'width': f.width, 'swatch_path': str(swatch_path), 'score': f.score}, ignore_index=True)

  df.to_pickle(output_path_str)


def get_processed_vids(output_path: Path):
  output_path_str = str(output_path)

  result = []
  if output_path.exists():
    df = pd.read_pickle(output_path_str)
    result = df['filename'].tolist()

  return result


def get_random_diffs(fake_frame_infos, real_frame_infos) -> List[RandomFrameDiff]:
  frame_rnd_diffs = []
  for image_fake_info in fake_frame_infos:
    image_fake, _, _, ndx, _ = image_fake_info

    image_real_info = real_frame_infos[ndx]
    image_real, _, _, _, _ = image_real_info

    o_height, o_width, _ = image_real.shape
    f_height, f_width, _ = image_fake.shape

    # if f_height * f_width > o_height * o_width:
    #   image_real = cv2.resize(image_real, (f_width, f_height), interpolation=cv2.INTER_NEAREST)
    # elif o_height * o_width > f_height * f_width:
    #   image_fake = cv2.resize(image_fake, (o_width, o_height), interpolation=cv2.INTER_NEAREST)

    # Choose random 244 x 244 from 1920, 1080
    height = 244
    width = 244
    max_x = o_width - width
    x_rnd = random.randint(1, max_x)

    max_y = o_height - height
    y_rnd = abs(random.randint(1, max_y))

    swatch_real = image_real[y_rnd:y_rnd + height, x_rnd:x_rnd + width]
    swatch_fake = image_fake[y_rnd:y_rnd + height, x_rnd:x_rnd + width]

    # logger.info(f"y_rnd: {y_rnd}: o_height: {o_height}; x_rnd: {x_rnd}; o_width: {o_width}")
    # logger.info(f"swatch_real: {swatch_real.shape}")

    score = get_ssim_score(swatch_fake, swatch_real)
    frame_rnd_diffs.append(RandomFrameDiff(swatch_fake, ndx, x_rnd, y_rnd, height, width, score))

  return frame_rnd_diffs


def get_ssim_score(image_fake, image_real):
  gray_real = cv2.cvtColor(image_real, cv2.COLOR_BGR2GRAY)
  gray_fake = cv2.cvtColor(image_fake, cv2.COLOR_BGR2GRAY)

  (score, diff_image) = structural_similarity(gray_real, gray_fake, full=True)

  # diff_image = (diff_image * 255).astype("uint8")
  # image_service.show_image(diff_image)

  print("SSIM: {}".format(score))

  return score
