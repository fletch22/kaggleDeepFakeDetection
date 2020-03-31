import os
from pathlib import Path

from config.LoggerFactory import LoggerFactory

PROJECT_PATH = Path(__name__).parent

LOGGING_FILE_PATH: str = "C:\\Users\\Chris\\workspaces\\data\\logs\\kaggleDeepFakeDection\\kaggleDeepFakeDection_1.log"

deepfake_folder_path = 'Kaggle Downloads\\deepfake-detection-challenge'

KAGGLE_DATA_PATH_C = f'C:\\{deepfake_folder_path}'
KAGGLE_DATA_PATH_D = f'D:\\{deepfake_folder_path}'
KAGGLE_DATA_PATH_E = f'E:\\{deepfake_folder_path}'
MP4_TRAIN_PATH = os.path.join(KAGGLE_DATA_PATH_D, "train_sample_videos")
SAMPLE_CSV_PATH = os.path.join(KAGGLE_DATA_PATH_D, 'sample_submission.csv')

TRAIN_PARENT_PATH_D = os.path.join(KAGGLE_DATA_PATH_D, "train")
TRAIN_PARENT_PATH_C = os.path.join(KAGGLE_DATA_PATH_C, "train")

TRAIN_DIR_PREFIX = 'dfdc_train_part_'

METADATA_FILENAME = 'metadata.json'

OUTPUT_PATH_D = os.path.join(KAGGLE_DATA_PATH_D, "output")
os.makedirs(OUTPUT_PATH_D, exist_ok=True)

OUTPUT_PATH_C = os.path.join(KAGGLE_DATA_PATH_C, "output")
os.makedirs(OUTPUT_PATH_C, exist_ok=True)

OUTPUT_PATH_E = os.path.join(KAGGLE_DATA_PATH_E, "output")
os.makedirs(OUTPUT_PATH_E, exist_ok=True)

SMALL_HEAD_OUTPUT_PATH = os.path.join(OUTPUT_PATH_D, "small_heads")
os.makedirs(SMALL_HEAD_OUTPUT_PATH, exist_ok=True)

SSIM_DIFFS_OUTPUT_PATH = os.path.join(OUTPUT_PATH_D, "ssim_diffs")
os.makedirs(SSIM_DIFFS_OUTPUT_PATH, exist_ok=True)

SSIM_RND_DIFFS_OUTPUT_PATH = os.path.join(OUTPUT_PATH_D, "ssim_rnd_diffs")
os.makedirs(SSIM_RND_DIFFS_OUTPUT_PATH, exist_ok=True)

SSIM_REALS_OUTPUT_PATH = os.path.join(OUTPUT_PATH_E, "ssim_reals")
os.makedirs(SSIM_REALS_OUTPUT_PATH, exist_ok=True)

SSIM_REALS_DATA_OUTPUT_PATH = Path(SSIM_REALS_OUTPUT_PATH, "data")
os.makedirs(SSIM_REALS_DATA_OUTPUT_PATH, exist_ok=True)

FACE_DET_PAR_PATH = Path(OUTPUT_PATH_D, "face_detections")
os.makedirs(FACE_DET_PAR_PATH, exist_ok=True)

FACE_DET_MAP_PATH = Path(OUTPUT_PATH_D, "face_detections_map")
os.makedirs(FACE_DET_MAP_PATH, exist_ok=True)

TRASH_PATH = os.path.join(OUTPUT_PATH_D, "trash")
os.makedirs(TRASH_PATH, exist_ok=True)

TINY_IMAGE_PATH = os.path.join(OUTPUT_PATH_D, "tiny_heads")
os.makedirs(TINY_IMAGE_PATH, exist_ok=True)

SAMPLE_IMAGES_PATH = os.path.join(KAGGLE_DATA_PATH_D, "images")
os.makedirs(SAMPLE_IMAGES_PATH, exist_ok=True)

WOMAN_PROFILE_IMAGE_PATH = os.path.join(SAMPLE_IMAGES_PATH, "woman_profile.jpg")

AUDIO_OUTPUT_PATH = os.path.join(OUTPUT_PATH_E, "audio")
os.makedirs(SMALL_HEAD_OUTPUT_PATH, exist_ok=True)

OUTPUT_METADATA_PATH = Path(OUTPUT_PATH_D, "metadata")
os.makedirs(OUTPUT_METADATA_PATH, exist_ok=True)

DF_ALL_METADATA_PATH = Path(str(OUTPUT_METADATA_PATH), 'df_all_metadata.pkl')

TEMP_OUTPUT_PAR_PATH = Path(OUTPUT_PATH_E, "tmp")
os.makedirs(TEMP_OUTPUT_PAR_PATH, exist_ok=True)

MERGED_SWATCH_PAR_PATH = Path(OUTPUT_PATH_C, 'merged')
os.makedirs(MERGED_SWATCH_PAR_PATH, exist_ok=True)

MERGED_SWATCH_DATA_PATH = Path(MERGED_SWATCH_PAR_PATH, 'data')
os.makedirs(MERGED_SWATCH_DATA_PATH, exist_ok=True)

MERGED_SWATCH_IMAGES_PATH = Path(MERGED_SWATCH_PAR_PATH, 'images')
os.makedirs(MERGED_SWATCH_IMAGES_PATH, exist_ok=True)

OUTPUT_VIRGIN_TEST_DATA_PAR_PATH = Path(OUTPUT_PATH_C, 'virgin_test_data')
os.makedirs(OUTPUT_VIRGIN_TEST_DATA_PAR_PATH, exist_ok=True)

OUTPUT_VIRGIN_TEST_DF_PAR_PATH = Path(OUTPUT_VIRGIN_TEST_DATA_PAR_PATH, 'dataframes')
os.makedirs(OUTPUT_VIRGIN_TEST_DF_PAR_PATH, exist_ok=True)

OUTPUT_VIRGIN_TEST_IMAGES_PAR_PATH = Path(OUTPUT_VIRGIN_TEST_DATA_PAR_PATH, 'images')
os.makedirs(OUTPUT_VIRGIN_TEST_IMAGES_PAR_PATH, exist_ok=True)

OUTPUT_MODEL_PAR_PATH = Path(OUTPUT_PATH_C, 'exported_models')
os.makedirs(OUTPUT_MODEL_PAR_PATH, exist_ok=True)

OUTPUT_LEARNER_PAR_PATH = Path(OUTPUT_MODEL_PAR_PATH, 'learners')
OUTPUT_LEARNER_PAR_PATH.mkdir(exist_ok=True)
OUTPUT_LEARNER_CNN_PATH = Path(OUTPUT_LEARNER_PAR_PATH, 'cnn_learner.pkl')

def get_train_batch_path(index: int):
  return os.path.join(TRAIN_PARENT_PATH_D, f"{TRAIN_DIR_PREFIX}{index}")

create_logger = LoggerFactory().create_logger
