import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import config
from services import image_service

logger = config.create_logger(__name__)


# https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py
def get_face_infos(image):
  face_locations = face_recognition.face_locations(image)

  padding_top_pct = 105
  padding_bottom_pct = 160
  padding_sides_pct = 50

  face_infos = []

  for face_location in face_locations:
    # Print the location of each face in this image
    top, right, bottom, left = face_location

    height = bottom - top
    padding_top = int((height * (padding_top_pct / 100)) / 2)
    padding_bottom = int((height * (padding_bottom_pct / 100)) / 2)
    padding_horiz = int(((right - left) * (padding_sides_pct / 100)) / 2)

    logger.info(f"pv: {padding_top}")

    top -= padding_top
    bottom += padding_bottom
    right += padding_horiz
    left -= padding_horiz

    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]

    face_infos.append((face_image, top, right, bottom, left))

  return face_infos


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

