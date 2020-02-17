from pathlib import Path

from services import file_service
import pandas as pd

class FaceDetectionIterator:
  def __init__(self, output_parent_path: Path):
    files = file_service.walk_to_path(output_parent_path, filename_endswith=".pkl")
    self.files = files

  def __len__(self):
    return len(self.files)

  def __getitem__(self, ndx: int):
    return pd.read_pickle(self.files[ndx])
