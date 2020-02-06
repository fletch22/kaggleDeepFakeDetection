from pathlib import Path

from torch.utils.data import Dataset

from BatchData import BatchData
from services import batch_data_loader_service, video_service


class BlazeDataSet(Dataset):
  def __init__(self, batch_index: int, vid_index: int):
    batch_data: BatchData = batch_data_loader_service.load_batch(batch_index)
    self.vid_path: Path = batch_data.get_candidate_file_path(vid_index)

    self.image_infos = video_service.process_all_video_frames(self.vid_path, None, None)

  def __len__(self):
    return video_service.get_num_frames(self.vid_path)

  def __getitem__(self, idx):
    img, _, _, _, _ = self.image_infos[0]
    return img
