from pathlib import Path

import cv2 as cv
from cv2 import cv2

import config

logger = config.create_logger(__name__)


def get_video_capture(video_file_path: Path):
  file_path = str(video_file_path)
  return cv.VideoCapture(file_path)


def get_single_image_from_vid(video_file_path: Path, frame_index: int = 0):
  cap = None
  try:
    cap = get_video_capture(video_file_path)
    cap.set(cv.cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, image = cap.read()

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
  finally:
    if cap is not None:
      cap.release()

  # matplotlib.use('TkAgg')
  # plt.interactive(False)

  return image


def process_all_video_frames(video_file_path: Path, fnProcess):
  cap = None
  results = []

  logger.info("About to get video.")
  try:
    cap = get_video_capture(video_file_path)
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    logger.info("About to process frames in video.")
    for frame_index in range(num_frames):
      logger.info(f"Processing frame {frame_index}.")
      success, image = cap.read()
      image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

      results.append(fnProcess(image, frame_index, video_file_path))

  finally:
    if cap is not None:
      cap.release()


  return results


def get_num_frames(video_file_path: Path) -> int:
  v_cap = None
  try:
    v_cap = cv.cv2.VideoCapture(str(video_file_path))
    num_frames = int(v_cap.get(cv.CAP_PROP_FRAME_COUNT))
    logger.info(f"Found {num_frames} fames.")
  finally:
    if v_cap is not None:
      v_cap.release()
  return num_frames
