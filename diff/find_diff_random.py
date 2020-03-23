import random
from pathlib import Path
from typing import List

import pandas as pd
from cv2 import cv2
from skimage.metrics import structural_similarity

import config
from diff.DiffSink import DiffSink
from diff.FaceFinder import FaceFinder
from diff.FaceSquare import FaceSquare
from diff.RandomFrameDiff import RandomFrameDiff
from services import video_service, face_recog_service
from services.RedisService import RedisService
from util import list_utils
from util.BatchData import BatchData
from util.BatchDataRow import BatchDataRow

logger = config.create_logger(__name__)

LEARNING_HEIGHT = 244
LEARNING_WIDTH = 244


class Point(object):
  def __init__(self, x: int, y: int):
    self.x = x
    self.y = y


class VideoPair():
  def __init__(self, vid_path_real: Path, vid_path_fake: Path):
    self.vid_path_real = vid_path_real
    self.vid_path_fake = vid_path_fake


class RandomVidDiffs():
  def __init__(self, vid_path: Path, diffs: List[RandomFrameDiff]):
    self.vid_path = vid_path
    self.diffs = diffs


def process_all_diffs(batch_data: BatchData, diff_sink: DiffSink, output_parent_path: Path, max_total_process: int = None, max_process_per_video: int = None):
  fakes_filtered: List[BatchDataRow] = BatchData.convert(batch_data.get_fakes())

  logger.info(f'Found {len(fakes_filtered)} number of fake videos in batch.')

  vid_pair_list = []
  logger.info(f'About to collect fake and real video pairs ...')
  for ndx, batch_row_data in enumerate(fakes_filtered):
    original_filename = batch_row_data.original_filename

    vid_path_fake: Path = batch_row_data.video_path
    vid_path_real = batch_data.get_vid_path(original_filename)

    vid_pair_list.append(VideoPair(vid_path_real, vid_path_fake))
  logger.info(f'Done collecting video pairs.')

  fakes_chunked = list_utils.chunks(vid_pair_list, 2)
  chunked_list = list(fakes_chunked)

  redis_service = RedisService()

  cum_processed = 0
  for chunk in chunked_list:
    if max_total_process is not None and cum_processed > max_total_process:
      break

    vid_diffs = []
    for vid_pair in chunk:
      is_max_processed = diff_sink.is_max_frames_processed(vid_pair.vid_path_fake, max_process_per_video)
      if is_max_processed:
        logger.info(f'Skipping video \'{vid_pair.vid_path_fake.name}\'. Already processed.')
        continue
      diff: RandomVidDiffs = process_diff(redis_service, diff_sink, vid_pair, max_process_per_video=max_process_per_video)
      vid_diffs.append(diff)
      cum_processed += len(diff.diffs)

    persist(diff_sink, vid_diffs, output_parent_path, max_swatches=max_total_process, randomize=True)

  return cum_processed


def get_unproc_fakes(batch_data, output_path):
  batch_row_data_list: List[BatchDataRow] = BatchData.convert(batch_data.df_metadata)
  processed_fakes = get_processed_vids(output_path)
  filtered = [f for f in batch_row_data_list if f.filename not in processed_fakes]

  logger.info(f"Skipping {len(batch_row_data_list) - len(filtered)} already processed files. Will process {len(filtered)} files.")

  return filtered


def process_diff(redis_service: RedisService, diff_sink: DiffSink, vp: VideoPair, max_process_per_video: int = None) -> RandomVidDiffs:
  face_finder = FaceFinder.load(redis_service, vp.vid_path_fake)

  fake_frame_infos = video_service.process_all_video_frames(vp.vid_path_fake, max_process=None)
  real_frame_infos = video_service.process_all_video_frames(vp.vid_path_real, max_process=None)

  diffs = get_random_diffs(face_finder, diff_sink, fake_frame_infos, real_frame_infos, max_process_per_video)

  return RandomVidDiffs(vp.vid_path_fake, diffs)


def persist(diff_sink: DiffSink, diffs: List[RandomVidDiffs], output_par_path: Path, max_swatches: int = None, randomize: bool = False) -> int:
  swatch_par_path = Path(str(output_par_path), 'swatches')
  if not swatch_par_path.exists():
    swatch_par_path.mkdir()

  cumu_swatches = 0
  for d in diffs:
    if max_swatches != None and cumu_swatches > max_swatches:
      break
    vp: Path = d.vid_path
    image_dir_path = Path(str(swatch_par_path), vp.stem)
    if not image_dir_path.exists():
      image_dir_path.mkdir()

    random_frame_diffs: List[RandomFrameDiff] = d.diffs

    if randomize is True:
      random.shuffle(random_frame_diffs)

    for f in random_frame_diffs:
      score_str = str(round(f.score, 5) * 100000)

      swatch_path = Path(str(image_dir_path), f'{f.frame_index}_{score_str}.png')
      img_fixed = cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB)
      cv2.imwrite(str(swatch_path), img_fixed)

      diff_sink.append(vp, f, swatch_path)
      cumu_swatches += 1

  diff_sink.persist()

  return cumu_swatches


def ensure_df(output_path):
  output_path_str = str(output_path)

  if output_path.exists():
    df = pd.read_pickle(output_path_str)
  else:
    df = pd.DataFrame(columns=['filename', 'path', 'frame_index', 'x', 'y', 'height', 'width', 'swatch_path', 'score'])

  return df


def get_processed_vids(output_path: Path):
  output_path_str = str(output_path)

  result = []
  if output_path.exists():
    df = pd.read_pickle(output_path_str)
    result = df['filename'].tolist()

  return result


def does_intersect_with_face(face_finder: FaceFinder, frame_index: int, l1: Point, r1: Point, height: int, width: int):
  frame_faces: List[FaceSquare] = face_finder.get_frame_faces(frame_index)

  result = False
  for ff in frame_faces:
    bottom, left, right, top = face_recog_service.adjust_face_boundary(ff.ymax, ff.xmin, ff.xmax, ff.ymin, width, height)
    l2 = Point(left, top)
    r2 = Point(right, bottom)

    # NOTE: 2020-03-04: Test if random square corner is inside face square: Quadrant 1 and 3
    if l2.x < l1.x < r2.x and ((l2.y < l1.y < r2.y) or (l2.y < r1.y < r2.y)):
      return True

    # NOTE: 2020-03-04: Quadrant 2 and 4
    if (l2.x < r1.x < r2.x) and ((l2.y < l1.y < r2.y) or (l2.y < r1.y < r2.y)):
      return True

    # NOTE: 2020-03-04: Test if face corner is inside random square: Quadrant 1 and 3
    if l1.x < l2.x < r1.x and ((l1.y < l2.y < r1.y) or (l1.y < r2.y < r1.y)):
      return True

    # NOTE: 2020-03-04: Quadrant 2 and 4
    if (l1.x < r2.x < r1.x) and ((l1.y < l2.y < r1.y) or (l1.y < r2.y < r1.y)):
      return True

  return result


def get_random_diffs(face_finder: FaceFinder, diff_sink: DiffSink, fake_frame_infos, real_frame_infos, max_process_per_video: int = None) -> List[RandomFrameDiff]:
  frame_rnd_diffs = []

  random.shuffle(fake_frame_infos)

  cum_diffed = None
  for image_fake_info in fake_frame_infos:
    image_fake, _, _, frame_index, vid_path = image_fake_info

    if cum_diffed is None:
      cum_diffed = diff_sink.get_num_frames_processed(vid_path)

    if max_process_per_video is not None and cum_diffed > max_process_per_video:
      break

    if diff_sink.is_frame_processed(vid_path, frame_index=frame_index):
      logger.info("Found item already processed. Moving on ...")
      continue

    image_real_info = real_frame_infos[frame_index]
    image_real, _, _, _, _ = image_real_info

    # Choose random 244 x 244 from 1920, 1080
    swatch_fake, swatch_real, x_rnd, y_rnd = get_swatch_pair(face_finder, frame_index, image_fake, image_real)

    if swatch_fake is None or swatch_real is None or x_rnd is None or y_rnd is None:
      continue

    score = get_ssim_score(swatch_fake, swatch_real)
    frame_rnd_diffs.append(RandomFrameDiff(swatch_fake, frame_index, x_rnd, y_rnd, LEARNING_HEIGHT, LEARNING_WIDTH, score))
    cum_diffed += 1

  return frame_rnd_diffs


def get_swatch_pair(face_finder, frame_index, image_fake, image_real):
  o_height, o_width, _ = image_real.shape
  f_height, f_width, _ = image_fake.shape

  x_rnd = None
  y_rnd = None

  swatch_fake = None
  swatch_real = None

  if o_height > LEARNING_HEIGHT and o_width > LEARNING_WIDTH:
    x_rnd, y_rnd = get_random_coords(face_finder, frame_index, o_height, o_width)
    if x_rnd is not None and y_rnd is not None:
      swatch_real = image_real[y_rnd:y_rnd + LEARNING_HEIGHT, x_rnd:x_rnd + LEARNING_WIDTH]
      swatch_fake = image_fake[y_rnd:y_rnd + LEARNING_HEIGHT, x_rnd:x_rnd + LEARNING_WIDTH]

  return swatch_fake, swatch_real, x_rnd, y_rnd


def get_random_coords(face_finder: FaceFinder, frame_index, o_height, o_width):
  max_x = o_width - LEARNING_WIDTH
  max_y = o_height - LEARNING_HEIGHT

  x_rnd = None
  y_rnd = None

  does_intersect = True
  try_count = 0
  max_try_count = 100
  while does_intersect is True:
    x_rnd = random.randint(1, max_x)
    y_rnd = random.randint(1, max_y)
    does_intersect = does_intersect_with_face(face_finder=face_finder, frame_index=frame_index, l1=Point(x_rnd, y_rnd), r1=Point(x_rnd + LEARNING_WIDTH, y_rnd + LEARNING_HEIGHT), height=o_height, width=o_width)
    try_count += 1
    if try_count > max_try_count:
      x_rnd = None
      y_rnd = None
      logger.info("Giving up trying to find random frame. Exceeded try limit.")
      break

  return x_rnd, y_rnd


def get_ssim_score(image_fake, image_real):
  gray_real = cv2.cvtColor(image_real, cv2.COLOR_BGR2GRAY)
  gray_fake = cv2.cvtColor(image_fake, cv2.COLOR_BGR2GRAY)

  (score, diff_image) = structural_similarity(gray_real, gray_fake, full=True)

  # diff_image = (diff_image * 255).astype("uint8")
  # image_service.show_image(diff_image)

  print("SSIM: {}".format(score))

  return score
