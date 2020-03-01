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
from services import video_service
from services.RedisService import RedisService
from util import list_utils
from util.BatchData import BatchData
from util.BatchDataRow import BatchDataRow

logger = config.create_logger(__name__)


class TwoDCoords(object):
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


def get_diffs(batch_data: BatchData, output_parent_path: Path, max_total_process: int = None, max_process_per_video: int = None):
  fakes_filterd: List[BatchDataRow] = BatchData.convert(batch_data.get_fakes())

  vid_pair_list = []
  for ndx, batch_row_data in enumerate(fakes_filterd):
    original_filename = batch_row_data.original_filename

    vid_path_fake: Path = batch_row_data.video_path
    vid_path_real = batch_data.get_vid_path(original_filename)

    vid_pair_list.append(VideoPair(vid_path_real, vid_path_fake))

  fakes_chunked = list_utils.chunks(vid_pair_list, 1)
  chunked_list = list(fakes_chunked)
  diff_sink = DiffSink(output_parent_path)

  redis_service = RedisService()

  cum_processed = 0
  for chunk in chunked_list:
    if max_total_process is not None and cum_processed > max_total_process:
      break

    vid_diffs = []
    for vid_pair in chunk:
      diff: RandomVidDiffs = process_diff(redis_service, diff_sink, vid_pair, max_process_per_video=max_process_per_video)
      vid_diffs.append(diff)
      cum_processed += len(diff.diffs)

    persist(diff_sink, vid_diffs, output_parent_path, max_swatches=max_total_process, randomize=True)


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

      if diff_sink.is_processed(vp, f.frame_index):
        continue

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


def does_intersect_with_face(face_finder: FaceFinder, frame_index: int, top_left_corner: TwoDCoords, bottom_right_corner: TwoDCoords):
  result = False

  frame_faces: List[FaceSquare] = face_finder.get_frame_faces(frame_index)

  for ff in frame_faces:
    if (ff.xmin < top_left_corner.x < ff.xmax
      and ff.ymin < top_left_corner.y < ff.ymax) \
      or (ff.xmin < bottom_right_corner.x < ff.xmax
          and ff.ymin < top_left_corner.y < ff.ymax) \
      or (ff.xmin < top_left_corner.x < ff.xmax
          and ff.ymin < bottom_right_corner.y < ff.ymax) \
      or (ff.xmin < bottom_right_corner.x < ff.xmax
          and ff.ymin < bottom_right_corner.y < ff.ymax):
      result = True
      break

  return result


def get_random_diffs(face_finder: FaceFinder, diff_sink: DiffSink, fake_frame_infos, real_frame_infos, max_process_per_video: int = None) -> List[RandomFrameDiff]:
  frame_rnd_diffs = []

  cum_diffed = 0
  for image_fake_info in fake_frame_infos:
    if max_process_per_video is not None and cum_diffed > max_process_per_video:
      break

    image_fake, _, _, frame_index, vid_path = image_fake_info

    if diff_sink.is_processed(vid_path, frame_index=frame_index):
      logger.info("Found item already processed. Moving on ...")
      continue

    image_real_info = real_frame_infos[frame_index]
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
    max_y = o_height - height

    x_rnd = None
    y_rnd = None

    does_intersect = True
    while (does_intersect):
      x_rnd = random.randint(1, max_x)
      y_rnd = abs(random.randint(1, max_y))
      does_intersect = does_intersect_with_face(face_finder=face_finder, frame_index=frame_index, top_left_corner=TwoDCoords(x_rnd, y_rnd), bottom_right_corner=TwoDCoords(x_rnd + width, y_rnd + width))
      if does_intersect:
        logger.info("Randomly chosen swatch intersects with face. Will attempt to choose another swatch.")

    swatch_real = image_real[y_rnd:y_rnd + height, x_rnd:x_rnd + width]
    swatch_fake = image_fake[y_rnd:y_rnd + height, x_rnd:x_rnd + width]

    score = get_ssim_score(swatch_fake, swatch_real)
    frame_rnd_diffs.append(RandomFrameDiff(swatch_fake, frame_index, x_rnd, y_rnd, height, width, score))
    cum_diffed += 1

  return frame_rnd_diffs


def get_ssim_score(image_fake, image_real):
  gray_real = cv2.cvtColor(image_real, cv2.COLOR_BGR2GRAY)
  gray_fake = cv2.cvtColor(image_fake, cv2.COLOR_BGR2GRAY)

  (score, diff_image) = structural_similarity(gray_real, gray_fake, full=True)

  # diff_image = (diff_image * 255).astype("uint8")
  # image_service.show_image(diff_image)

  print("SSIM: {}".format(score))

  return score
