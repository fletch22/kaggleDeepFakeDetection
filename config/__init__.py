import os
from pathlib import Path

from config.LoggerFactory import LoggerFactory

PROJECT_PATH = Path(__name__).parent

LOGGING_FILE_PATH: str = "C:\\Users\\Chris\\workspaces\\data\\logs\\kaggleDeepFakeDection\\kaggleDeepFakeDection_1.log"

KAGGLE_DATA_PATH_D = 'D:\\Kaggle Downloads\\deepfake-detection-challenge'
KAGGLE_DATA_PATH_C = "C:\\Kaggle Downloads\\deepfake-detection-challenge"
MP4_TRAIN_PATH = os.path.join(KAGGLE_DATA_PATH_D, "train_sample_videos")
SAMPLE_CSV_PATH = os.path.join(KAGGLE_DATA_PATH_D, 'sample_submission.csv')

TRAIN_PARENT_PATH_D = os.path.join(KAGGLE_DATA_PATH_D, "train")
TRAIN_PARENT_PATH_C = os.path.join(KAGGLE_DATA_PATH_C, "train")

TRAIN_DIR_PREFIX = 'dfdc_train_part_'

METADATA_FILENAME = 'metadata.json'

OUTPUT_PATH = os.path.join(KAGGLE_DATA_PATH_D, "output")
SMALL_HEAD_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "small_heads")
os.makedirs(SMALL_HEAD_OUTPUT_PATH, exist_ok=True)

FACE_DET_PAR_PATH = Path(OUTPUT_PATH, "face_detections")
os.makedirs(FACE_DET_PAR_PATH, exist_ok=True)

TRASH_PATH = os.path.join(OUTPUT_PATH, "trash")
os.makedirs(TRASH_PATH, exist_ok=True)

TINY_IMAGE_PATH = os.path.join(OUTPUT_PATH, "tiny_heads")
os.makedirs(TINY_IMAGE_PATH, exist_ok=True)

SAMPLE_IMAGES_PATH = os.path.join(KAGGLE_DATA_PATH_D, "images")
os.makedirs(SAMPLE_IMAGES_PATH, exist_ok=True)

WOMAN_PROFILE_IMAGE_PATH = os.path.join(SAMPLE_IMAGES_PATH, "woman_profile.jpg")

def get_train_batch_path(index: int):
  return os.path.join(TRAIN_PARENT_PATH_D, f"{TRAIN_DIR_PREFIX}{index}")


AUDIO_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "audio")
os.makedirs(SMALL_HEAD_OUTPUT_PATH, exist_ok=True)


create_logger = LoggerFactory().create_logger
