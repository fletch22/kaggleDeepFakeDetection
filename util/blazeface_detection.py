from pathlib import Path
from typing import List

from cv2 import cv2
from torch.utils.data.dataloader import DataLoader

import config
from services import file_service
from util.BlazeDataSet import BlazeDataSet
from util.FaceDetection import FaceDetection
from util.blazeface import BlazeFace

logger = config.create_logger(__name__)


def batch_detect(blaze_dataloader, blazeface: BlazeFace):
  logger.info("About to batch detect in all subframes.")
  all_video_detections = []
  for i_batch, sample_batched in enumerate(blaze_dataloader):
    # Change batch to include all 3 cropped images.
    h = blazeface.predict_on_batch(sample_batched.detach().numpy())

    faces = [item.detach().cpu().numpy() for item in h]
    all_video_detections.extend(faces)

  return all_video_detections


def save_cropped_blazeface_image(all_video_detections, blaze_dataset: BlazeDataSet, output_path: Path):
  logger.info(f"Saving cropped faces from video.")
  for detections in all_video_detections:

    # logger.info(f"Number subframe detections: {len(detections)}")
    for head_ndx, d in enumerate(detections):
      vid_path = blaze_dataset.vid_path
      frame_index = d['frame_index']
      image_orig = blaze_dataset.originals[frame_index]
      height_orig, width_orig, _ = image_orig.shape
      coords = d['coords']
      ymin = coords['ymin']
      ymax = coords['ymax']
      xmin = coords['xmin']
      xmax = coords['xmax']

      xmax_pad, xmin_pad, ymax_pad, ymin_pad = blaze_dataset.pad_face_crop(xmax, xmin, ymax, ymin, height_orig, width_orig)
      image_orig_cropped_face = image_orig[ymin_pad:ymax_pad, xmin_pad:xmax_pad]

      output_image_set_path = Path(output_path, f"{vid_path.stem}")
      if not output_image_set_path.exists():
        output_image_set_path.mkdir()
      face_path = Path(output_image_set_path, f"{frame_index}_{head_ndx}.png")

      # logger.info(str(face_path))
      # image_service.show_image(image_orig_cropped_face)

      if face_path.exists():
        face_path.unlink()
      image_converted = cv2.cvtColor(image_orig_cropped_face, cv2.COLOR_BGR2RGB)
      cv2.imwrite(str(face_path), image_converted)


def get_video_paths(parent_folder_paths: List[str]):
  video_paths = []
  for par_path in parent_folder_paths:
    files = file_service.walk_to_path(par_path, filename_endswith=".mp4")
    video_paths.extend(files)

  return video_paths


def multi_batch_detection(video_paths: List[Path], max_rows_to_process: int = None, output_parent_path: Path = None):
  blazeface = BlazeFace()

  face_det = FaceDetection(output_parent_path=output_parent_path)

  row_count = 1
  for vp in video_paths:
    blaze_dataset = BlazeDataSet(vp)

    if len(blaze_dataset.originals) == 0:
      logger.info(f"No data in {blaze_dataset.vid_path}")
      continue

    blaze_dataloader = DataLoader(blaze_dataset, batch_size=300, shuffle=False, num_workers=0)

    all_video_detections = batch_detect(blaze_dataloader, blazeface)

    merged_vid_detections = blaze_dataset.merge_sub_frame_detections(all_video_detections)

    # save_cropped_blazeface_image(merged_vid_detections, blaze_dataset, output_folder_path)
    face_det.add_row(vp, merged_vid_detections)
    face_det.persist()

    if max_rows_to_process is not None and row_count >= max_rows_to_process:
      break
    row_count += 1


if __name__ == '__main__':
  output_parent_path = config.FACE_DET_PAR_PATH
  parent_folder_paths = [config.TRAIN_PARENT_PATH_C, config.TRAIN_PARENT_PATH_D]
  video_paths = FaceDetection.get_video_paths(parent_folder_paths=parent_folder_paths, output_parent_path=output_parent_path)

  multi_batch_detection(video_paths=video_paths, output_parent_path=output_parent_path)
