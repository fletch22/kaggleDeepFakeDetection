from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt

import config

logger = config.create_logger(__name__)


def get_video_capture(video_file_path: Path):
  file_path = str(video_file_path)
  return cv.VideoCapture(file_path)


def get_image_from_vid(video_file_path: Path, frame_index: int = 0):
  cap = get_video_capture(video_file_path)
  cap.set(cv.cv2.CAP_PROP_POS_FRAMES, frame_index)
  success, image = cap.read()

  image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
  cap.release()

  # matplotlib.use('TkAgg')
  # plt.interactive(False)

  return image



