from pathlib import Path

import config
from diff.FaceFinder import FaceFinder
from services.RedisService import RedisService
from util import blazeface_detection
from util.FaceDetection import FaceDetection
from util.FaceDetectionIterator import FaceDetectionIterator

logger = config.create_logger(__name__)


# Detect face locations in all frames.
def pipeline_stage_1():
  output_parent_path = config.FACE_DET_PAR_PATH
  parent_folder_paths = [config.TRAIN_PARENT_PATH_C, config.TRAIN_PARENT_PATH_D]
  video_paths = FaceDetection.get_video_paths(parent_folder_paths=parent_folder_paths, output_parent_path=output_parent_path)

  blazeface_detection.multi_batch_detection(video_paths=video_paths, output_parent_path=output_parent_path)

  df_iterator: FaceDetectionIterator = FaceDetection.get_dataframes_iterator(output_parent_path=output_parent_path)

  for df in df_iterator:
    logger.info(f"Number of rows from first pickle file: {df.shape[0]}")
    break

# Copy face locations to Redis
def pipeline_stage_2():
  fdi = FaceDetectionIterator(config.FACE_DET_PAR_PATH)

  redis_service = RedisService()

  for df_det in fdi:
    for _, row in df_det.iterrows():
      det = row['detections']
      vid_path = Path(row['path'])

      face_finder = FaceFinder(vid_path)
      face_finder.add_detections(det)

      logger.info(f'Writing to redis \'{vid_path.name}\'.')
      redis_service.write_binary(FaceFinder.get_redis_key(vid_path), face_finder)

if __name__ == '__main__':
    pipeline_stage_2()

