from datetime import datetime
from pathlib import Path
from unittest import TestCase

import config
from diff.SwatchShuttle import SwatchShuttle

logger = config.create_logger(__name__)

class TestSwatchShuttle(TestCase):

  def test_consolidate(self):
    # Arrange
    dt = str(datetime.utcnow().isoformat()).replace(':', '-').replace('.', '-')
    logger.info(dt)
    output_path = Path(config.SSIM_REALS_DATA_OUTPUT_PATH, f'cumulative_{dt}.pkl')

    # Act
    SwatchShuttle.consolidate_pickles(output_path)
    # Assert