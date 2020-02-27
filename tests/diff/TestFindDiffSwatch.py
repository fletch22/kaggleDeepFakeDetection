from pathlib import Path
from unittest import TestCase

import config
from diff import find_diff_swatch
from services import batch_data_loader_service, video_service
from util import pd_util
from util.BatchData import COL_VID_PATH, COL_ORIGINAL, COL_CANDIDATE, BatchData

logger = config.create_logger(__name__)


class TestFindDiffSwatch(TestCase):

  def test_spot_diff(self):
    # Arrange
    output_path = Path(config.SSIM_DIFFS_OUTPUT_PATH, 'ssim_diffs.pkl')
    batch_data = batch_data_loader_service.load_batch(3)

    max_process = None

    # Act
    diffs = find_diff_swatch.get_diffs(batch_data, output_path, max_process)

    # Assert

  def test_shoeme(self):
    batch_data = batch_data_loader_service.load_batch(0)

    candi_list = batch_data.df_metadata[COL_CANDIDATE].tolist()

    for ndx, row in batch_data.df_metadata.iterrows():
      path = Path(row[COL_VID_PATH])
      if not path.exists():
        raise Exception("foo")
