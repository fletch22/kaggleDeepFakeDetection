from pathlib import Path

import numpy as np
from cv2 import cv2

import config
from services import image_service
from util.BlazeDataSet import BlazeDataSet

logger = config.create_logger(__name__)


def batch_detect(blaze_dataloader, blazeface):
  logger.info("About to batch detect in all subframes.")
  all_video_detections = []
  for i_batch, sample_batched in enumerate(blaze_dataloader):
    # Change batch to include all 3 cropped images.
    h = blazeface.predict_on_batch(sample_batched.detach().numpy())

    # # Necessary for PyTorch tensor processing
    faces = [item.detach().cpu().numpy() for item in h]
    all_video_detections.extend(faces)

    # If no face is found the PyTorch tensor returned has shape [0, 17], otherwise [1, 17]
    # num_faces = np.sum([f.shape[0] for f in faces])
    # logger.info(f"Number of faces found: {num_faces}")

  return all_video_detections

def save_cropped_blazeface_image(all_video_detections, blaze_dataset: BlazeDataSet):
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

      output_image_set_path = Path(config.SMALL_HEAD_IMAGE_PATH, f"{vid_path.stem}")
      if not output_image_set_path.exists():
        output_image_set_path.mkdir()
      face_path = Path(output_image_set_path, f"{frame_index}_{head_ndx}.png")

      # logger.info(str(face_path))
      # image_service.show_image(image_orig_cropped_face)

      if face_path.exists():
        face_path.unlink()
      image_converted = cv2.cvtColor(image_orig_cropped_face, cv2.COLOR_BGR2RGB)
      cv2.imwrite(str(face_path), image_converted)

