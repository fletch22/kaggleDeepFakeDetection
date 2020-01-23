import os
from pathlib import Path
from typing import Tuple

import cv2 as cv
from cv2 import cv2

import config
from services import spark_service, face_recog_service

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


def process_all_video_frames_with_spark(video_file_path: Path):
  cap = None

  logger.info("About to get video.")
  try:
    cap = get_video_capture(video_file_path)
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    frame_infos = [(str(video_file_path), i) for i in range(num_frames)]

    face_data = spark_service.execute(frame_infos, process_one_frame_with_spark, num_slices=20)
  finally:
    if cap is not None:
      cap.release()

  return face_data


def process_one_frame_with_spark(frame_info: Tuple):
  video_file_path_str, index = frame_info
  video_file_path = Path(video_file_path_str)
  image = get_single_image_from_vid(video_file_path, index)

  face_data = face_recog_service.get_face_data(image, index, video_file_path)

  frame_index = face_data['frame_index']
  face_info_landmarks = face_data['face_info_landmarks']

  for head_index, fil in enumerate(face_info_landmarks):
    face_image = fil['face_image']
    face_path = os.path.join(config.SMALL_HEAD_IMAGE_PATH, f"{video_file_path.name}_{frame_index}_{head_index}.jpg")
    logger.info(face_path)

    face_image_converted = cv.cvtColor(face_image, cv.COLOR_BGR2RGB)
    cv2.imwrite(face_path, face_image_converted)

  return face_data
