from pathlib import Path

from cv2 import cv2
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim as ssim

import config

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


def get_image_differences(original_file_path: Path, fake_file_path: Path):
  image_o = cv2.imread(str(original_file_path))
  o_height, o_width, _ = image_o.shape

  image_f = cv2.imread(str(fake_file_path))
  f_height, f_width, _ = image_f.shape

  if f_height * f_width > o_height * o_width:
    image_o = cv2.resize(image_o, (f_width, f_height), interpolation=cv2.INTER_NEAREST)
  else:
    image_f = cv2.resize(image_f, (o_width, o_height), interpolation=cv2.INTER_NEAREST)

  image_o_conv = cv2.cvtColor(image_o, cv2.COLOR_BGR2RGB)
  image_f_conv = cv2.cvtColor(image_f, cv2.COLOR_BGR2RGB)

  return dict(original_image=image_o_conv, fake_image=image_f_conv, ssim=ssim(image_o, image_f, multichannel=True))
