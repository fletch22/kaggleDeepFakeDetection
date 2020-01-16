import os

from config.LoggerFactory import LoggerFactory

LOGGING_FILE_PATH: str = "C:\\Users\\Chris\\workspaces\\data\\logs\\kaggleDeepFakeDection\\kaggleDeepFakeDection_1.log"

KAGGLE_DATA_PATH = 'D:\\Kaggle Downloads\\deepfake-detection-challenge'
MP4_TRAIN_PATH = os.path.join(KAGGLE_DATA_PATH, "train_sample_videos")
SAMPLE_CSV_PATH = os.path.join(KAGGLE_DATA_PATH, 'sample_submission.csv')

TRAIN_PARENT_PATH = os.path.join(KAGGLE_DATA_PATH, "train")

TRAIN_DIR_PREFIX = 'dfdc_train_part_'

METADATA_FILENAME = 'metadata.json'


def get_train_batch_path(index: int):
  return os.path.join(TRAIN_PARENT_PATH, f"{TRAIN_DIR_PREFIX}{index}")


create_logger = LoggerFactory().create_logger
