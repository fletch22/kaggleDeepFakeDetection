from pathlib import Path
from typing import List, Dict

from cv2 import cv2, os
from mtcnn.mtcnn import MTCNN

import config
from util.BatchData import BatchData
from services import file_service, image_service, face_recog_service, video_service

logger = config.create_logger(__name__)


def get_single_face(image_path: Path):
  img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

  detector = MTCNN()
  return detector.detect_faces(img)


def get_faces_with_path(video_path_list: List[Path], output_path: Path, expand_frame: bool):
  video_info_list = []
  detector = MTCNN()
  for vid_path in video_path_list:
    video_info_list.extend(find_face(detector, vid_path, output_path, expand_frame=expand_frame))

  return video_info_list


def find_face(detector, video_path: Path, output_path: Path, expand_frame: bool):
  face_path_list = []

  cap = None

  try:
    cap = video_service.get_video_capture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_head_path = Path(os.path.join(str(output_path), f"{video_path.stem}"))
    os.makedirs(str(output_head_path), exist_ok=True)
    existing_frames = get_existing_frame_heads(output_head_path)

    logger.info(f"Processing '{video_path.name}' ...")
    for frame_index in range(num_frames):
      if frame_index in existing_frames:
        continue

      logger.info(f"Processing '{video_path.name}' frame {frame_index}.")
      success, image = cap.read()
      total_image_height, total_image_width, _ = image.shape

      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      fi = detector.detect_faces(image)

      for head_index, face in enumerate(fi):
        face_path = process_face(face, frame_index, head_index, image, total_image_height, total_image_width, output_head_path, expand_frame)
        face_path_list.append(face_path)

  finally:
    if cap is not None:
      cap.release()

  return face_path_list


def process_face(face, frame_index, head_index, image, total_image_height, total_image_width, output_path: Path, expand_frame: bool = True):
  # [{'box': [417, 326, 128, 184], 'confidence': 0.9998984336853027, 'keypoints': {'left_eye': (432, 395), 'right_eye': (470, 385), 'nose': (425, 426), 'mouth_left': (443, 475), 'mouth_right': (469, 467)}}]
  box = face['box']
  x = box[0]
  y = box[1]
  width = box[2]
  height = box[3]

  bottom = y + height
  left = x
  right = x + width
  top = y

  if expand_frame:
    bottom, left, right, top = face_recog_service.adjust_face_boundary(bottom, left, right, top, total_image_height, total_image_width)

  face_image = image[top:bottom, left:right]
  face_path = os.path.join(str(output_path), f"{frame_index}_{head_index}.png")
  face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

  # image_service.show_image(face_image, 'test')
  cv2.imwrite(face_path, face_image)

  return face_path


def get_small_head_dirpath(video_path: Path):
  return os.path.join(config.SMALL_HEAD_OUTPUT_PATH, f"{video_path.stem}")


def chunk_video_list(video_info_list):
  vid_info = video_info_list[0]
  number_buckets = 10
  chunks = []
  num_frames = vid_info['num_frames']
  bucket_size = num_frames // number_buckets
  frames = range(number_buckets)
  for i in frames:
    start_range = i * bucket_size
    end_range = start_range + bucket_size
    if end_range > num_frames - bucket_size:
      end_range = num_frames + 1
    chu = dict(video_path=str(vid_info['video_path']), start_range=start_range, end_range=end_range)
    chunks.append(chu)
  return chunks, number_buckets


def find_face_one_at_time(video_info: Dict):
  detector = MTCNN()

  video_path = Path(video_info['video_path'])
  start_range = video_info['start_range']
  end_range = video_info['end_range']

  face_path_list = []
  for frame_index in range(start_range, end_range):
    image, _, _ = video_service.get_single_image_from_vid(video_path, frame_index)
    logger.info(f"Processing '{video_path.name}' frame {frame_index}.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fi = detector.detect_faces(image)

    for index, face in enumerate(fi):
      # [{'box': [417, 326, 128, 184], 'confidence': 0.9998984336853027, 'keypoints': {'left_eye': (432, 395), 'right_eye': (470, 385), 'nose': (425, 426), 'mouth_left': (443, 475), 'mouth_right': (469, 467)}}]
      # logger.info(fi)
      box = face['box']
      x = box[0]
      y = box[1]
      width = box[2]
      height = box[3]

      bottom = y + height
      left = x
      right = x + width
      top = y

      bottom, left, right, top = face_recog_service.adjust_face_boundary(bottom, left, right, top)

      # You can access the actual face itself like this:
      face_image = image[top:bottom, left:right]
      # image_service.show_image(face_image, f"Frame {frame_index}, head {ndx}.")
      face_path = os.path.join(config.SMALL_HEAD_OUTPUT_PATH, f"{video_path.name}_{frame_index}_{index}.jpg")
      # face_image_converted = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
      cv2.imwrite(face_path, face_image)
      face_path_list.append(face_path)

  return True


def get_existing_frame_heads(vid_path: Path) -> List[int]:
  files = file_service.walk(str(vid_path))

  existing_frames = []
  for f in files:
    stem = Path(f).stem
    stem_parts = stem.split("_")
    if os.path.getsize(str(f)) == 0:
      logger.info(f"Removing file with zero (0) bytes. ({Path(f).name})")
      os.remove(f)
    else:
      existing_frames.append(int(stem_parts[0]))

  return existing_frames


def get_faces(batch_data: BatchData, output_path: Path, expand_frame: bool):
  num_videos = batch_data.size()
  vid_path_list: List[Path] = []
  for i in range(num_videos):
    logger.info(f"Getting {i}th video.")

    vid_path_list.append(batch_data.get_candidate_file_path(i))

  return get_faces_with_path(video_path_list=vid_path_list, output_path=output_path, expand_frame=expand_frame)


def find_tiny_faces(root_path: Path, output_path: Path):
  expand_frame = False
  detector = MTCNN()

  files = file_service.walk(root_path)
  face_path_list = []
  for ndx, f in enumerate(files):
    if ndx > 10:
      break
    file_path = Path(f)
    filename = file_path.stem
    filename_parts = filename.split("_")
    frame_index = filename_parts[0]
    head_index = filename_parts[1]
    image = cv2.imread(f)
    output_head_path = os.path.join(str(output_path), os.path.dirname(f))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_service.show_image(image_rgb, "test")
    # fi = detector.detect_faces(image_rgb)
    #
    # for _, face in enumerate(fi):
    #   face_path = process_mtcnn_face(face, frame_index, head_index, image, output_head_path, expand_frame)
    #   face_path_list.append(face_path)

  return face_path_list
