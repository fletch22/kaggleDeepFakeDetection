from typing import Dict


class FakeSwatchData():

  def __init__(self, frame: Dict):
    self.filename = frame['filename']
    self.path = frame['path']
    self.frame_index = frame['frame_index']
    self.height = frame['height']
    self.width = frame['width']
    self.x = frame['x']
    self.y = frame['y']
    self.swatch_path = frame['swatch_path']