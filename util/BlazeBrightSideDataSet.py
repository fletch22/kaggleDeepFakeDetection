from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from cv2 import cv2
from torch.utils.data import Dataset

import config
from services import video_service

logger = config.create_logger(__name__)


class BlazeBrightSideDataSet(Dataset):

  def __init__(self, vid_path: Path, max_process=None):
    self.resize_height = 128
    self.resize_width = 128

    self.coords_list = []
    self.originals = []
    self.coords_map = {}
    self.original_frame_info = {}

    if vid_path is None:
      raise Exception(f"Vid_path cannot be None. ")

    self.vid_path = vid_path

    video_service.process_all_video_frames(self.vid_path, self.get_image_meta, max_process)

  def get_image_meta(self, image, height, width, frame_index):
    frame = {
      "image": image,
      "last_heads_locs": []
    }

    self.original_frame_info[frame_index] = frame

  def __len__(self):
    return len(self.original_frame_info.keys())

  def __getitem__(self, frame_index) -> Dict:
    return self.original_frame_info[frame_index]

  def get_subframe_images_info(self, frame_index: int):
    frame = self.original_frame_info[frame_index]
    image = frame['image']
    height = image.shape[0]
    width = image.shape[1]

    sub_frame_coords = self.get_sub_frame_coords(height, width)

    image_list = []
    for coords in sub_frame_coords:
       image_list.append(self.get_resized_image_and_coords(image, coords))

    return image_list

  def get_resized_image_and_coords(self, image: np.ndarray, sub_frame_coords: Dict) -> Dict:
    ymin = sub_frame_coords['ymin']
    ymax = sub_frame_coords['ymax']
    xmin = sub_frame_coords['xmin']
    xmax = sub_frame_coords['xmax']
    image_cropped = image[ymin:ymax, xmin:xmax]

    img_resized = cv2.resize(image_cropped, (self.resize_height, self.resize_width), interpolation=cv2.INTER_AREA)

    return dict(image=img_resized, sub_frame_coords=sub_frame_coords, resize_dim=(self.resize_height, self.resize_width))

  def get_sub_frame_coords(self, height: int, width: int):
    sub_frame_coords = []

    sub_frame_coords.append(self.get_first_subframe(height, width))
    sub_frame_coords.append(self.get_second_subframe(height, width))
    sub_frame_coords.append(self.get_third_subframe(height, width))

    return sub_frame_coords

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

  def merge_sub_frame_detections(self, image_info_detections: List) -> List[dict]:

    frame_det_coords = self.convert_subframe_det_to_original_det(image_info_detections)

    return self.get_unique_faces(frame_det_coords)

    # logger.info(f"About to get unique faces in subframes.")
    #
    # det_map = self.convert_subframe_det_to_map(all_detections)
    #
    # all_frame_detections = []
    # for ndx, frame_index in enumerate(det_map.keys()):
    #   frame_detections = det_map[frame_index]
    #
    #   frame_det_coords = self.convert_subframe_det_to_original_det(frame_detections, frame_index)
    #
    #   all_frame_detections.append(self.get_unique_faces(frame_det_coords, frame_index))
    #
    # return all_frame_detections

  def convert_subframe_det_to_original_det(self, image_info_detections: List[Dict]) -> List:
    frame_det_coords = []
    for image_info_det in image_info_detections:
      image_info = image_info_det['image_info']
      sub_frame_coords = image_info['sub_frame_coords']
      frame_index = image_info_det['frame_index']
      face_detections = image_info_det['face_detections']

      coord_list = self.get_face_coord_on_original(face_detections, frame_index, sub_frame_coords)

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

  def get_unique_faces(self, frame_det_coords: List) -> List[Dict]:
    min_diff = 50
    unique_face_coords: List[Tuple[int, Any, Any]] = []
    for ndx, coords in enumerate(frame_det_coords):
      xmin = coords['xmin']
      ymin = coords['ymin']

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

        if is_unique:
          unique_face_coords.append((ndx, xmin, ymin))

    return [frame_det_coords[ndx] for (ndx, _, _) in unique_face_coords]

  def get_face_coord_on_original(self, detections_in_sub_frame: List, frame_index: int, coords: Dict) -> List:
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

      image_orig = self.original_frame_info[frame_index]['image']
      original_height, original_width, _ = image_orig.shape

      height_crop = o_ymax - o_ymin
      width_crop = o_xmax - o_xmin

      ymin_oface = int(ymin_frac * height_crop) + o_ymin
      ymax_oface = int(ymax_frac * height_crop) + o_ymin
      xmin_oface = int(xmin_frac * width_crop) + o_xmin
      xmax_oface = int(xmax_frac * width_crop) + o_xmin

      original_coords.append(dict(xmax=xmax_oface, xmin=xmin_oface, ymax=ymax_oface, ymin=ymin_oface))

    return original_coords
