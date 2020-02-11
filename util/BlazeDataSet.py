from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from cv2 import cv2
from torch.utils.data import Dataset

import config
from BatchData import BatchData
from services import batch_data_loader_service, video_service

logger = config.create_logger(__name__)


class BlazeDataSet(Dataset):
  originals = []
  coords_map = {}
  coords_list = []

  def __init__(self, batch_index: int = None, vid_index: int = None, max_process=None, vid_path: Path = None):
    self.resize_height = 128
    self.resize_width = 128

    if vid_path is None:
      batch_data: BatchData = batch_data_loader_service.load_batch(batch_index)
      self.vid_path: Path = batch_data.get_candidate_file_path(vid_index)
    else:
      self.vid_path = vid_path

    logger.info(f"Getting {self.vid_path.name}.")

    # self.image_metas = video_service.process_all_video_frames(self.vid_path, self.get_image_meta, max_process)
    video_service.process_all_video_frames(self.vid_path, self.get_image_meta, max_process)

  def __len__(self):
    return len(self.coords_list)

  def __getitem__(self, sub_frame_index) -> np.ndarray:
    sub_frame_info = self.coords_list[sub_frame_index]
    frame_index = sub_frame_info['frame_index']
    image_original = self.originals[frame_index]
    # image_info = self.image_metas[frame_index]

    # sub_frame_coords_list = image_info['sub_frame_coords']
    # sub_frame_images = []
    # for sub_frame_coords in sub_frame_coords_list:
    #   sub_frame_images.append(self.get_subframe(image_original, sub_frame_coords=sub_frame_coords, frame_index=frame_index))
    # return sub_frame_images

    return self.get_subframe_image(image_original, sub_frame_coords=sub_frame_info['sub_frame_coords'], frame_index=frame_index)

  def get_subframe_image(self, image_original: np.ndarray, sub_frame_coords: Dict, frame_index: int) -> np.ndarray:
    ymin = sub_frame_coords['ymin']
    ymax = sub_frame_coords['ymax']
    xmin = sub_frame_coords['xmin']
    xmax = sub_frame_coords['xmax']
    image_cropped = image_original[ymin:ymax, xmin:xmax]

    img_resized = cv2.resize(image_cropped, (self.resize_height, self.resize_width), interpolation=cv2.INTER_AREA)
    # sub_frame_info = dict(image=img_resized, sub_frame_coords=sub_frame_coords, frame_index=frame_index, resize_dim=(self.resize_height, self.resize_width))
    # return sub_frame_info

    return img_resized

  def get_image_meta(self, image: np.ndarray, height: int, width: int, frame_index: int):
    self.originals.append(image)

    # Get 3 subframes: 1st position, 2nd pos., 3rd pos.
    sub_frame_coords = []

    sub_frame_coords = []
    if frame_index in self.coords_map.keys():
      sub_frame_coords = self.coords_map[frame_index]
    else:
      self.coords_map[frame_index] = sub_frame_coords

    # sub_frame_coords.append(self.get_first_subframe(height, width))
    # sub_frame_coords.append(self.get_second_subframe(height, width))
    # sub_frame_coords.append(self.get_third_subframe(height, width))

    coord_info_1 = dict(frame_index=frame_index, sub_frame_coords=self.get_first_subframe(height, width))
    sub_frame_coords.append(coord_info_1)

    coord_info_2 = dict(frame_index=frame_index, sub_frame_coords=self.get_second_subframe(height, width))
    sub_frame_coords.append(coord_info_2)

    coord_info_3 = dict(frame_index=frame_index, sub_frame_coords=self.get_third_subframe(height, width))
    sub_frame_coords.append(coord_info_3)

    self.coords_list.extend(sub_frame_coords)

    # return self.get_transformation_meta(sub_frame_coords, frame_index)

  def get_transformation_meta(self, sub_frame_coords: List[Dict], frame_index: int):
    return dict(sub_frame_coords=sub_frame_coords, frame_index=frame_index,
                video_path=self.vid_path,
                resize_dim=(self.resize_height, self.resize_width))

  def get_first_subframe(self, height, width) -> Dict:
    if width > height:
      ymin = 0
      ymax = height
      xmin = 0
      xmax = height
    else:
      ymin = 0
      ymax = width
      xmin = 0
      xmax = width
    return dict(xmax=xmax, xmin=xmin, ymax=ymax, ymin=ymin)

  def get_second_subframe(self, height, width) -> Dict:
    if width > height:
      ymin = 0
      ymax = height
      xmin = (width - height) // 2
      xmax = xmin + height
    else:
      ymin = (height - width) // 2
      ymax = ymin + width
      xmin = 0
      xmax = width
    return dict(xmax=xmax, xmin=xmin, ymax=ymax, ymin=ymin)

  def get_third_subframe(self, height, width) -> Dict:
    if width > height:
      ymin = 0
      ymax = height
      xmin = width - height
      xmax = width
    else:
      ymin = height - width
      ymax = height
      xmin = 0
      xmax = width
    return dict(xmax=xmax, xmin=xmin, ymax=ymax, ymin=ymin)

  def get_face_images_in_subframe(self, detections_in_sub_frame, sub_frame_index: int) -> List[Dict]:
    coords_info = self.coords_list[sub_frame_index]
    frame_index = coords_info['frame_index']

    vid_path = self.vid_path
    sub_frame_coords = coords_info['sub_frame_coords']
    o_ymin = sub_frame_coords['ymin']
    o_ymax = sub_frame_coords['ymax']
    o_xmin = sub_frame_coords['xmin']
    o_xmax = sub_frame_coords['xmax']

    # logger.info(f"El 0: {detections_in_frame}: video_path: {str(vid_path)}; frame_index: {frame_index}")

    image_faces_found = []
    for face in detections_in_sub_frame:
      if face.shape[0] == 0:
        image_faces_found.append(dict(frame_index=frame_index, image=None, video_path=vid_path))
        continue

      ymin_frac = face[0]
      xmin_frac = face[1]
      ymax_frac = face[2]
      xmax_frac = face[3]

      image_orig = self.originals[frame_index]
      original_height, original_width, _ = image_orig.shape

      height_cropped = o_ymax - o_ymin
      width_cropped = o_xmax - o_xmin

      ymin_oface = int(ymin_frac * height_cropped) + o_ymin
      ymax_oface = int(ymax_frac * height_cropped) + o_ymin

      xmin_oface = int(xmin_frac * width_cropped) + o_xmin
      xmax_oface = int(xmax_frac * width_cropped) + o_xmin

      xmax_oface, xmin_oface, ymax_oface, ymin_oface = self.pad_face_crop(xmax_oface, xmin_oface, ymax_oface, ymin_oface, original_height, original_width)
      image_orig_cropped_face = image_orig[ymin_oface:ymax_oface, xmin_oface:xmax_oface]
      image_faces_found.append(dict(frame_index=frame_index, image=image_orig_cropped_face, video_path=vid_path, xmin=xmin_oface, ymin=ymin_oface))

    return image_faces_found

  def pad_face_crop(self, x_max, x_min, y_max, y_min, original_height, original_width):
    margin_vert = .45
    margin_horiz = .12

    height = y_max - y_min
    y_min = int(y_min - (height * margin_vert))
    if y_min < 0: y_min = 0

    y_max = int(y_max + (height * margin_vert))
    if y_max > original_height: y_max = original_height

    width = x_max - x_min
    x_min = int(x_min - (width * margin_horiz))
    if x_min < 0: x_min = 0

    x_max = int(x_max + (width * margin_horiz))
    if x_max > original_width: x_max = original_width

    return x_max, x_min, y_max, y_min

  def merge_sub_frame_detections(self, all_detections: List) -> List[List]:
    logger.info(f"About to get unique faces in subframes.")

    det_map = self.convert_subframe_det_to_map(all_detections)

    all_frame_detections = []
    for ndx, frame_index in enumerate(det_map.keys()):
      frame_detections = det_map[frame_index]

      frame_det_coords = self.convert_subframe_det_to_original_det(frame_detections, frame_index)

      # logger.info(f"num: {len(frame_det_coords)}")

      all_frame_detections.append(self.get_unique_faces(frame_det_coords, frame_index))

    return all_frame_detections

  def convert_subframe_det_to_original_det(self, frame_detections, frame_index) -> List:
    frame_det_coords = []
    for fd in frame_detections:
      coord_index = fd["coord_index"]
      sub_frame_detections = fd['sub_frame_detections']

      coord_list = self.get_face_coord_on_original(sub_frame_detections, coord_index)

      # logger.info(f"frame index: {frame_index}; coord_index: {coord_index}; coord_list: {coord_list}")

      frame_det_coords.extend(coord_list)

    return frame_det_coords

  def convert_subframe_det_to_map(self, all_detections) -> Dict:
    det_map = {}
    for ndx, d in enumerate(all_detections):
      coords_list = self.coords_list[ndx]
      frame_index = coords_list['frame_index']

      frame_detections = []
      if frame_index in det_map.keys():
        frame_detections = det_map[frame_index]
      else:
        det_map[frame_index] = frame_detections

      frame_detections.append(dict(coord_index=ndx, sub_frame_detections=d))

    return det_map

  def get_unique_faces(self, frame_det_coords: List, frame_index: int) -> List[Dict]:
    min_diff = 50
    unique_face_coords: List[Tuple[int, Any, Any]] = []
    for ndx, coords in enumerate(frame_det_coords):
      xmin = coords['xmin']
      ymin = coords['ymin']

      # logger.info(f"frame_index: {frame_index}; xmin: {xmin}; ymin: {ymin}")

      if len(unique_face_coords) == 0:
        unique_face_coords.append((ndx, xmin, ymin))
      else:
        # Are any coords in unique close? If yes then skip.
        is_unique = True
        for in_ndx, (u_ndx, u_xmin, u_ymin) in enumerate(unique_face_coords):
          x_diff = abs(xmin - u_xmin)
          y_diff = abs(ymin - u_ymin)

          if x_diff < min_diff and y_diff < min_diff:
            is_unique = False
            break

        # logger.info(f"x_diff: {x_diff}; y_diff: {y_diff}")
        # logger.info(f"x_diff: {x_diff}; y_diff: {y_diff}")
        if is_unique:
          unique_face_coords.append((ndx, xmin, ymin))

    return [dict(coords=frame_det_coords[ndx], frame_index=frame_index) for (ndx, _, _) in unique_face_coords]

  def get_face_coord_on_original(self, detections_in_sub_frame: List, coord_index: int) -> List:
    c = self.coords_list[coord_index]
    frame_index = c['frame_index']
    coords = c['sub_frame_coords']

    o_ymin = coords['ymin']
    o_ymax = coords['ymax']
    o_xmin = coords['xmin']
    o_xmax = coords['xmax']

    original_coords = []
    for face in detections_in_sub_frame:
      if face.shape[0] == 0:
        original_coords.append(dict(xmax=None, xmin=None, ymax=None, ymin=None))
        continue

      ymin_frac = face[0]
      xmin_frac = face[1]
      ymax_frac = face[2]
      xmax_frac = face[3]

      image_orig = self.originals[frame_index]
      original_height, original_width, _ = image_orig.shape

      height_crop = o_ymax - o_ymin
      width_crop = o_xmax - o_xmin

      ymin_oface = int(ymin_frac * height_crop) + o_ymin
      ymax_oface = int(ymax_frac * height_crop) + o_ymin
      xmin_oface = int(xmin_frac * width_crop) + o_xmin
      xmax_oface = int(xmax_frac * width_crop) + o_xmin

      original_coords.append(dict(xmax=xmax_oface, xmin=xmin_oface, ymax=ymax_oface, ymin=ymin_oface))

    return original_coords
