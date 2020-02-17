import os
from pathlib import Path

from cv2 import cv2
from pandas import DataFrame
from tensorflow_core.python.ops.image_ops_impl import ssim

import config
from util.BatchData import BatchData
from services import file_service, image_service


def get_face_diffs(batch_data: BatchData, max_diffs: int = 1):
  df: DataFrame = batch_data.df_metadata

  df_fakes = df[df['label'] == 'FAKE']

  small_dir_path = config.SMALL_HEAD_OUTPUT_PATH

  diffs = []
  for ndx, row in df_fakes.iterrows():
    if len(diffs) > max_diffs:
      break
    fake_filename = row['candidate_filename']
    original_filename = row['original_filename']

    orig_dirname = Path(original_filename).stem
    fake_dirname = Path(fake_filename).stem

    orig_dir_path = os.path.join(small_dir_path, orig_dirname)
    fake_dir_path = os.path.join(small_dir_path, fake_dirname)

    if os.path.exists(orig_dir_path) and os.path.exists(fake_dir_path):
      orig_files = file_service.walk(orig_dir_path)
      orig_file_info = get_file_info(orig_files)

      for of in orig_file_info:
        file_path = of['file_path']
        frame_index = of['frame_index']
        head_index = of['head_index']

        file_to_find = os.path.join(fake_dir_path, f"{frame_index}_{head_index}.png")
        fake_path = Path(file_to_find)
        if fake_path.exists():
          image_diffs = get_image_differences(file_path, fake_path)
          ssim = image_diffs['ssim']
          if ssim > 0.93:
            image_service.show_image(image_diffs['original_image'], f'original {ssim}')
            image_service.show_image(image_diffs['fake_image'], f'fake {ssim}')
          head_diff = dict(ssim=ssim, original_path=file_path, fake_path=fake_path)
          diffs.append(head_diff)

  return diffs


def get_file_info(files):
  file_info = []
  for f in files:
    file_path = Path(f)
    stem_parts = file_path.stem.split("_")
    frame_index = stem_parts[0]
    head_index = stem_parts[1]
    file_info.append(dict(frame_index=frame_index, head_index=head_index, file_path=file_path))

  return file_info


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
