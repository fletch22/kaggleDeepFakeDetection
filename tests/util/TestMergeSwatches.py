from pathlib import Path
from unittest import TestCase

import pandas as pd

import config
from diff import find_real_swatches
from diff.DiffSink import DiffSink
from models import cnn_regression
from services import file_service
from util import merge_swatches

logger = config.create_logger(__name__)

real_filename_full = 'df_r.pkl'
fake_filename_full = 'df_r.pkl'

real_filename_tiny = 'df_r_tiny.pkl'
fake_filename_tiny = 'df_r_tiny.pkl'


class TestMergeSwatches(TestCase):

  def test_get_data(self):
    # Arrange
    df_f, df_r = merge_swatches.getTempDataframes()
    # df_f: pd.DataFrame = DiffSink.load_history()
    # df_f.to_pickle(str(tmp_output_path_f))

    # Act
    cnn_regression.get_data_from_list([df_f, df_r])

    # Assert
    assert (df_r is not None)
    assert (df_f is not None)

  def test_merge_test_data(self):
    # Arrange
    df_f, df_r = merge_swatches.getTempDataframes()

    logger.info(f'Real: {df_r.columns}')
    logger.info(f'Fake: {df_f.columns}')

  def test_map_path_to_score(self):
    # Arrange
    destination = Path(config.TEMP_OUTPUT_PAR_PATH, 'merged')
    destination.mkdir(exist_ok=True)

    file_service.delete_files(destination)

    max_process = 10

    # Act
    map = merge_swatches.merge_real_and_fake(destination, max_process)

    # Assert
    assert(len(map.keys()) == 10)

    for k in map.keys():
      logger.info(f'Key: {k}; Entry: {map[k]}')

  def test_merge(self):
    # Arrange
    destination = Path(config.TEMP_OUTPUT_PAR_PATH, 'merged')
    destination.mkdir(exist_ok=True)

    file_service.delete_files(destination)

    max_process = 10

    # Act
    map = merge_swatches.merge_real_and_fake(destination, max_process)
