from unittest import TestCase

import config
from diff import find_real_swatches
from services import batch_data_loader_service

logger = config.create_logger(__name__)


class TestFindRealSwatches(TestCase):

  def test_load(self):
    # Arrange
    filename = 'aaagqkcdis.mp4'
    df = batch_data_loader_service.read_all_metadata_as_df()

    # Act
    real_video_path = find_real_swatches.get_real_video_path(filename, df)

    # Assert
    assert (real_video_path.exists)

  def test_get_real(self):
    # Arrange
    # batch_data_loader_service.save_all_metadata_to_single_df()
    erase_history = False
    max_proc_swatches = 745860

    # Act
    find_real_swatches.get_real(erase_history=erase_history, max_proc_swatches=max_proc_swatches)

    # Assert

  def test_find_real_from_fake(self):
    # Arrange
    # D:\Kaggle Downloads\deepfake-detection-challenge\train\dfdc_train_part_11\aabdogagch.mp4
    # 2020-03-07 16:26:59,562 - diff.find_real_swatches - INFO - Will output real swatch to: E:\Kaggle Downloads\deepfake-detection-challenge\output\ssim_reals\swatches\ytoamycnx
    fake_filename = 'aabdogagch.mp4'
    df_metadata = batch_data_loader_service.read_all_metadata_as_df()

    # Act
    real_video_path = find_real_swatches.get_real_video_path(fake_filename, df_metadata)

    # Assert
    logger.info(f'Real video: {real_video_path}')
