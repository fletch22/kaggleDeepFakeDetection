from pandas import Series

COL_CANDIDATE = "candidate_filename"
COL_ORIGINAL = "original_filename"
COL_VID_PATH = "vid_path"
COL_FACE_DETECTIONS = "face_detections"
COL_FAKE_OR_REAL = "label"
COL_SPLIT = "split"


class BatchDataRow():

  def __init__(self, row: Series):
    self.filename = row[COL_CANDIDATE]
    self.original_filename = row[COL_ORIGINAL]
    self.video_path = row[COL_VID_PATH]
    self.split = row[COL_SPLIT]


