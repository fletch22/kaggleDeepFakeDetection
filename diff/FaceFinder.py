from __future__ import annotations
from pathlib import Path
from typing import Dict, List

from diff.FaceSquare import FaceSquare
from services.RedisService import RedisService


class FaceFinder():
  def __init__(self, vid_path: Path):
    self.vid_path = vid_path
    self.frame_detections = {}

  @staticmethod
  def get_redis_key(vid_path: Path):
    return vid_path.name

  def add_detections(self, detections: List[List[Dict]]):
    # Example: [[{'coords': {'xmax': 1069, 'xmin': 887, 'ymax': 454, 'ymin': 272}, 'frame_index': 0}], ...
    fd = self.frame_detections

    for faces_in_frame in detections:
      for face in faces_in_frame:
        frame_index_key = str(face['frame_index'])
        if frame_index_key in fd.keys():
          face_coords = fd[frame_index_key]
        else:
          face_coords = []
          fd[frame_index_key] = face_coords

        coords = face['coords']
        face_square = FaceSquare(coords['xmax'], coords['xmin'], coords['ymax'], coords['ymin'])
        face_coords.append(face_square)

  def get_frame_faces(self, frame_index) -> List[FaceSquare]:
    result = []
    key = str(frame_index)
    if key in self.frame_detections.keys():
      result = self.frame_detections[key]

    return result

  @staticmethod
  def load(redis_service: RedisService, vid_path: Path) -> FaceFinder:
    key = FaceFinder.get_redis_key(vid_path)
    face_finder: FaceFinder = redis_service.read_binary(key)
    if face_finder is None:
      raise Exception(f'Encountered problem with getting a face finder from redis. FaceFinder \'{key}\' could not be found.')
    return face_finder