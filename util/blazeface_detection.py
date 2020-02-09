from pathlib import Path

import numpy as np
from cv2 import cv2

import config
from util.BlazeDataSet import BlazeDataSet

logger = config.create_logger(__name__)


def batch_detect(blaze_dataloader, blazeface):
  all_video_detections = []
  for i_batch, sample_batched in enumerate(blaze_dataloader):
    h = blazeface.predict_on_batch(sample_batched.detach().numpy())

    # Necessary for PyTorch tensor processing
    faces = [item.detach().cpu().numpy() for item in h]
    all_video_detections.extend(faces)

    # If no face is found the PyTorch tensor returned has shape [0, 17], otherwise [1, 17]
    num_faces = np.sum([f.shape[0] for f in faces])

    logger.info(f"Number of faces found: {num_faces}")

  return all_video_detections


def save_cropped_blazeface_image(all_video_detections, blaze_dataset: BlazeDataSet):
  for ndx in range(len(all_video_detections)):
    image_list = blaze_dataset.get_face_snapshot(all_video_detections, ndx)
    logger.info(f"found {len(image_list)} faces in single frame.")
    for head_index, i_dict in enumerate(image_list):
      video_path = i_dict['video_path']
      frame_index = i_dict['frame_index']
      face_image = i_dict['image']
      if face_image is not None:
        output_image_set_path = Path(config.SMALL_HEAD_IMAGE_PATH, f"{video_path.stem}")
        if not output_image_set_path.exists():
          output_image_set_path.mkdir()
        face_path = Path(output_image_set_path, f"{frame_index}_{head_index}.png")
        # logger.info(str(face_path))
        # image_service.show_image(face_image)
        if face_path.exists():
          face_path.unlink()
        image_converted = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(face_path), image_converted)
