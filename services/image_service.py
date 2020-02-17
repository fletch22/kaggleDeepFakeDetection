from pathlib import Path

from matplotlib import pyplot as plt

import config
from util.BatchData import BatchData
from services import batch_data_loader_service, video_service

logger = config.create_logger(__name__)


def show_image(image, title: str = None):
  fig = plt.figure(frameon=False)
  ax = fig.add_subplot(1, 1, 1)
  ax.axis('off')
  ax.title.set_text(title)
  ax.xaxis.set_visible(False)
  ax.yaxis.set_visible(False)
  ax.imshow(image)
  plt.grid(False)
  plt.show()


def pick_image(batch_index: int, video_index: int, frame_index: int):
  batch_data: BatchData = batch_data_loader_service.load_batch(batch_index)
  vid_path: Path = batch_data.get_candidate_file_path(video_index)

  return video_service.get_single_image_from_vid(vid_path, frame_index)
