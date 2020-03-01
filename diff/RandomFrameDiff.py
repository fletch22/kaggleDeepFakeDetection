import numpy as np


class RandomFrameDiff():
  def __init__(self, image: np.ndarray, frame_index: int, x: int, y: int, height: int, width: int, score: float):
    self.image = image
    self.frame_index = frame_index
    self.x = x
    self.y = y
    self.height = height
    self.width = width
    self.score = score
