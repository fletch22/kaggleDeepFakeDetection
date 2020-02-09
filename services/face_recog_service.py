import os
from pathlib import Path
from typing import List, Tuple, Dict, Union

import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from cv2 import data

import config

logger = config.create_logger(__name__)


# https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py
def get_face_infos(image, total_image_height, total_image_width) -> List[Tuple]:
  face_locations = face_recognition.face_locations(image)
  face_infos = []

  for face_location in face_locations:
    # Print the location of each face in this image
    top, right, bottom, left = face_location

    bottom, left, right, top = adjust_face_boundary(bottom, left, right, top, total_image_height, total_image_width)

    logger.info(face_location)
    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]

    face_infos.append((face_image, top, right, bottom, left))

  return face_infos


def adjust_face_boundary(bottom, left, right, top, total_image_width, total_image_height):
  padding_top_pct = 105
  padding_bottom_pct = 160
  padding_sides_pct = 50
  height = bottom - top

  padding_top = int((height * (padding_top_pct / 100)) / 2)
  padding_bottom = int((height * (padding_bottom_pct / 100)) / 2)
  padding_horiz = int(((right - left) * (padding_sides_pct / 100)) / 2)

  top -= padding_top
  top = 0 if top < 0 else top
  padded_bottom = bottom + padding_bottom
  bottom = bottom if padded_bottom > total_image_height else padded_bottom
  padded_right = right + padding_horiz
  right = right if padded_right > total_image_width else padded_right
  left -= padding_horiz
  left = 0 if left < 0 else left

  return bottom, left, right, top


def add_face_lines(image):
  face_landmarks_list = face_recognition.face_landmarks(image)

  # https://github.com/ageitgey/face_recognition/blob/master/examples/find_facial_features_in_picture.py
  # face_landmarks_list
  pil_image = Image.fromarray(image)
  d = ImageDraw.Draw(pil_image)

  for face_landmarks in face_landmarks_list:
    # Print the location of each facial feature in this image
    for facial_feature in face_landmarks.keys():
      print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

    # Let's trace out each facial feature in the image with a line!
    for facial_feature in face_landmarks.keys():
      d.line(face_landmarks[facial_feature], width=3)

  # revert to original image type
  return np.array(pil_image)


def get_face_data(image, height, width, frame_index: int, file_path: Path) -> Dict[str, Union[List[Dict[str, Union[object, List]]], int, str]]:
  face_infos = get_face_infos(image, height, width)
  faces_list = []

  if len(face_infos) == 0:
    face_infos = get_haar_face_data(image, height, width)

  for fi in face_infos:
    face_image, _, _, _, _ = fi
    face_landmarks_list = face_recognition.face_landmarks(face_image)
    face_info_landmarks = dict(face_image=face_image, face_landmarks_list=face_landmarks_list)
    faces_list.append(face_info_landmarks)

  if len(faces_list) == 0:
    logger.info(f"No face found for frame {frame_index} in '{file_path.name}'.")

  return dict(face_info_landmarks=faces_list, frame_index=frame_index, file_path=file_path)


def get_haar_face_data(image, total_image_height, total_image_width):
  # Gets the name of the image file (filename) from sys.argv
  cascPath = os.path.join(data.haarcascades, 'haarcascade_profileface.xml')

  faceCascade = cv2.CascadeClassifier(cascPath)

  # image = cv2.imread(str(imagePath))
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # The face or faces in an image are detected
  # This section requires the most adjustments to get accuracy on face being detected.
  faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(1, 1),
    flags=cv2.CASCADE_SCALE_IMAGE
  )

  # This draws a green rectangle around the faces detected
  face_infos = []
  for (x, y, w, h) in faces:
    top = y
    bottom = y + h
    left = x
    right = x + w

    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    bottom, left, right, top = adjust_face_boundary(bottom, left, right, top, total_image_height, total_image_width)

    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]

    face_infos.append((face_image, top, right, bottom, left))

  return face_infos
