from pathlib import Path

from cv2 import cv2
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from BatchData import BatchData
from services import batch_data_loader_service, video_service

trans = transforms.Compose([
  transforms.ToPILImage(),
  transforms.Resize((128, 128)),
  transforms.ToTensor()])

resize_height = 128
resize_width = 128


class BlazeDataSet(Dataset):
  originals = []

  def __init__(self, batch_index: int, vid_index: int, max_process=None):
    batch_data: BatchData = batch_data_loader_service.load_batch(batch_index)
    self.vid_path: Path = batch_data.get_candidate_file_path(vid_index)

    self.image_infos = video_service.process_all_video_frames(self.vid_path, self.transform, max_process)

  def __len__(self):
    return len(self.image_infos)

  def __getitem__(self, idx):
    img, _, _, _, _ = self.image_infos[0]

    return img

  def transform(self, image, height, width, frame_index, video_file_path):
    self.originals.append(image)
    center_offset = ((width - height) // 2)
    image_cropped = image[:, center_offset:height + center_offset]
    return cv2.resize(image_cropped, (resize_height, resize_width), interpolation=cv2.INTER_AREA), resize_height, resize_width, frame_index, video_file_path
