import os
from pathlib import Path
from typing import Tuple

from cv2 import cv2

import config
from services import video_service, face_recog_service

logger = config.create_logger(__name__)


def process_one_frame_with_spark(frame_info: Tuple):
  video_file_path_str, index = frame_info
  video_file_path = Path(video_file_path_str)
  image, height, width = video_service.get_single_image_from_vid(video_file_path, index)

  face_data = face_recog_service.get_face_data(image, height, width, index, video_file_path)

  frame_index = face_data['frame_index']
  face_info_landmarks = face_data['face_info_landmarks']

  for head_index, fil in enumerate(face_info_landmarks):
    face_image = fil['face_image']
    output_image_set_path = os.path.join(config.SMALL_HEAD_OUTPUT_PATH, f"{video_file_path.stem}")
    face_path = os.path.join(output_image_set_path, f"{frame_index}_{head_index}.png")
    logger.info(face_path)

    # face_image_converted = cv.cvtColor(face_image, cv.COLOR_BGR2RGB)
    cv2.imwrite(face_path, face_image)

  return face_data
