from pathlib import Path
from typing import Tuple, List

import pandas as pd
from cv2 import cv2
from pandas import DataFrame

import config
from diff.DiffSink import DiffSink
from diff.FakeSwatchData import FakeSwatchData
from diff.SwatchShuttle import SwatchShuttle
from services import batch_data_loader_service, video_service, pickle_service, file_service
from util.BatchData import COL_CANDIDATE, COL_ORIGINAL, COL_VID_PATH

logger = config.create_logger(__name__)


def get_real(erase_history=False, max_proc_swatches=None):

  reals_output_par_path: Path = Path(config.SSIM_REALS_OUTPUT_PATH)

  df: DataFrame = DiffSink.load_history()
  df_grouped = df.groupby('filename')

  df_metadata = batch_data_loader_service.read_all_metadata_as_df()

  df_process_history = handle_history(erase_history)

  df_grouped_history = df_process_history.groupby('filename')

  history_map = {}
  for filename, df_group in df_grouped_history:
    history_map[filename] = df_group['frame_index'].tolist()

  all_shuttles = []
  total_processed = 0
  for fake_filename, group in df_grouped:
    real_video_path = get_real_video_path(fake_filename, df_metadata)
    records = group.to_dict(orient='records')

    all_frames = []
    for r in records:
      fsd = FakeSwatchData(r)

      if df_process_history is not None:
        filename = real_video_path.name
        if filename in history_map.keys():
          frame_indexes = history_map[filename]
          if fsd.frame_index in frame_indexes:
            total_processed += 1
            logger.info(f'Skipping record. Already exists. Skipped or processed {total_processed}.')
            continue

      all_frames.append(fsd)

    if len(all_frames) > 0:
      swatch_shuttle = SwatchShuttle(all_frames, real_video_path=real_video_path, output_par_path=reals_output_par_path)
      all_shuttles.append(swatch_shuttle)

  # spark_service.execute(all_shuttles, spark_process, num_slices=2)


  for ndx, s in enumerate(all_shuttles):
    total_processed += spark_process(s)
    if max_proc_swatches is not None and max_proc_swatches < total_processed:
      break


def handle_history(erase_history):
  df_process_history = None
  if erase_history:
    # Erase history.
    _erase_history()
  else:
    df_process_history = load_history()
  return df_process_history


def get_real_video_path(filename: str, df: pd.DataFrame):
  real_filename = df[df[COL_CANDIDATE] == filename][COL_ORIGINAL].tolist()[0]
  real_video_path = df[df[COL_CANDIDATE] == real_filename][COL_VID_PATH].tolist()[0]

  return real_video_path


def spark_process(swatch_shuttle: SwatchShuttle):
  fake_video_path = None
  frames_to_process = []
  for ndx, fsd in enumerate(swatch_shuttle):
    if fake_video_path is None:
      fake_video_path = fsd.path
    frames_to_process.append(fsd.frame_index)

  results = video_service.process_specific_video_frames(swatch_shuttle.real_video_path, frames_to_process)

  logger.info(f'Number of frames: {len(results)}')

  logger.info(f'From fake video path: {fake_video_path}')

  for ndx, fsd in enumerate(swatch_shuttle):
    frame_index = fsd.frame_index
    height = fsd.height
    width = fsd.width
    x = fsd.x
    y = fsd.y

    swatch_path = swatch_shuttle.get_swatch_path(ndx)
    swatch_shuttle.save_data(ndx, swatch_path)
    logger.info(f'Will output real swatch to: {swatch_path}')
    # Get original image
    image = find_image_from_process_results(results, frame_index)

    # crop to swatch
    image = image[y:(y + height), x:(x + width)]

    # save to file
    image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(swatch_path), image_converted)

  swatch_shuttle.persist_to_disk()

  return len(results)


def find_image_from_process_results(results: List[Tuple], frame_index_to_find: int):
  for (image, _, _, frame_index, _) in results:
    if frame_index == frame_index_to_find:
      return image


def load_history(max_pickles=None):
  logger.info("About to load real swatch process history ...")
  df, _ = pickle_service.concat_pickled_dataframes(config.SSIM_REALS_DATA_OUTPUT_PATH, max_pickles)
  return df


def _erase_history():
  if config.SSIM_REALS_DATA_OUTPUT_PATH.exists():
    files = file_service.walk_to_path(config.SSIM_REALS_DATA_OUTPUT_PATH, filename_endswith='.pkl')
    for f in files:
      f.unlink()


if __name__ == '__main__':
  erase_history = False
  max_proc_swatches = 1745860

  get_real(erase_history=erase_history, max_proc_swatches=max_proc_swatches)
