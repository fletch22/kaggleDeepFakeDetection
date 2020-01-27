import os

from config.LoggerFactory import LoggerFactory

LOGGING_FILE_PATH: str = "C:\\Users\\Chris\\workspaces\\data\\logs\\kaggleDeepFakeDection\\kaggleDeepFakeDection_1.log"

KAGGLE_DATA_PATH = 'D:\\Kaggle Downloads\\deepfake-detection-challenge'
MP4_TRAIN_PATH = os.path.join(KAGGLE_DATA_PATH, "train_sample_videos")
SAMPLE_CSV_PATH = os.path.join(KAGGLE_DATA_PATH, 'sample_submission.csv')

TRAIN_PARENT_PATH = os.path.join(KAGGLE_DATA_PATH, "train")

TRAIN_DIR_PREFIX = 'dfdc_train_part_'

METADATA_FILENAME = 'metadata.json'

OUTPUT_PATH = os.path.join(KAGGLE_DATA_PATH, "output")
SMALL_HEAD_IMAGE_PATH = os.path.join(OUTPUT_PATH, "small_heads")
os.makedirs(SMALL_HEAD_IMAGE_PATH, exist_ok=True)

TRASH_PATH = os.path.join(OUTPUT_PATH, "trash")
os.makedirs(TRASH_PATH, exist_ok=True)

TINY_IMAGE_PATH = os.path.join(OUTPUT_PATH, "tiny_heads")
os.makedirs(TINY_IMAGE_PATH, exist_ok=True)

SAMPLE_IMAGES_PATH = os.path.join(KAGGLE_DATA_PATH, "images")
os.makedirs(SAMPLE_IMAGES_PATH, exist_ok=True)

WOMAN_PROFILE_IMAGE_PATH = os.path.join(SAMPLE_IMAGES_PATH, "woman_profile.jpg")

def get_train_batch_path(index: int):
  return os.path.join(TRAIN_PARENT_PATH, f"{TRAIN_DIR_PREFIX}{index}")


create_logger = LoggerFactory().create_logger
