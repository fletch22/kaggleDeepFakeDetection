from pathlib import Path
from random import shuffle

import config
from diff import find_diff_random, find_real_swatches
from diff.DiffSink import DiffSink
from diff.FaceFinder import FaceFinder
from pipelines.DecorateDf import DecorateDf
from services import batch_data_loader_service, file_service
from services.RedisService import RedisService
from util import blazeface_detection, merge_swatches
from util.BatchData import BatchData
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


# Create swatches
def pipeline_stage_3():
  # Arrange
  output_par_path = Path(config.SSIM_RND_DIFFS_OUTPUT_PATH)

  files = batch_data_loader_service.get_metadata_json_files()
  shuffle(files)

  # batch_data = batch_data_loader_service.load_batch(3)
  max_process = 300000
  max_process_per_video = 30
  diff_sink = DiffSink(output_par_path)

  num_processed = 0
  for f in files:
    if num_processed > max_process:
      break
    batch_data: BatchData = batch_data_loader_service.load_batch_from_path(f)
    num_processed += find_diff_random.process_all_diffs(batch_data, diff_sink, output_par_path, max_total_process=max_process, max_process_per_video=max_process_per_video)

  logger.info(f'Total Processed: {num_processed}.')


# Save all metadata to single file
def pipeline_stage_4():
  batch_data_loader_service.save_all_metadata_to_single_df()


def pipeline_stage_5():
  erase_history = False
  max_proc_swatches = 745860

  find_real_swatches.get_real(erase_history=erase_history, max_proc_swatches=max_proc_swatches)

# Copy all image to the same folder'
def pipeline_stage_6():
  destination = config.MERGED_SWATCH_PAR_PATH
  file_service.delete_files(destination)

  max_process = 1000000
  map = merge_swatches.merge_real_and_fake(destination, max_process)

def pipeline_stage_7():
  # learn
  root_output = config.OUTPUT_PATH_C
  pipeline_dirname = 'decorate_df'
  pipeline = DecorateDf(root_output=root_output, pipeline_dirname=pipeline_dirname)

  result = pipeline.start(pct_train=80, pct_test=5)

  logger.info(f'Result: {result}')

if __name__ == '__main__':
  pipeline_stage_7()
