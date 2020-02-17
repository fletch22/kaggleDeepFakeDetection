from pathlib import Path
from unittest import TestCase

import config
from util.BatchData import BatchData
from services import audio_service, batch_data_loader_service

logger = config.create_logger(__name__)


class TestAudioService(TestCase):

  def test_get_audio_file(self):
    # Arrange
    batch_data: BatchData = batch_data_loader_service.load_batch(8)
    vid_path: Path = batch_data.get_candidate_file_path(0)
    start_millis = 1000
    end_millis = 5000

    # Act
    clip, sample_rate = audio_service.get_audio_clip_from_video(vid_path=vid_path, start_milli=start_millis, end_milli=end_millis)

    logger.info(f"Sample rate: {sample_rate}")

    # Assert

  def test_display_audio_chart(self):
    # Arrange
    batch_data: BatchData = batch_data_loader_service.load_batch(8)
    vid_path: Path = batch_data.get_candidate_file_path(0)
    clip, sample_rate = audio_service.get_audio_clip_from_video(vid_path=vid_path, start_milli=1000, end_milli=5000)

    # Act
    audio_service.display_chart(clip, sample_rate)

    # Assert
