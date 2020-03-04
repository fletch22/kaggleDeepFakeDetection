from pathlib import Path
from unittest import TestCase

from pandas import DataFrame

import config
from diff.DiffSink import DiffSink

logger = config.create_logger(__name__)

class TestDiffSink(TestCase):

  def test_get_persisted(self):
    # Arrange
    output_par_path: Path = Path(config.SSIM_RND_DIFFS_OUTPUT_PATH, 'data')

    # Act
    df: DataFrame = DiffSink.get_persisted(output_par_path)

    # Assert
    logger.info(f'Num rows: {df.shape[0]}')

    assert(df.shape[0] > 0)
