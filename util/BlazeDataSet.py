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

  def __init__(self, batch_index: int = None, vid_index: int = None, max_process=None, vid_path: Path=None):
    if vid_path is None:
      batch_data: BatchData = batch_data_loader_service.load_batch(batch_index)
      self.vid_path: Path = batch_data.get_candidate_file_path(vid_index)
    else:
      self.vid_path = vid_path

    self.image_infos = video_service.process_all_video_frames(self.vid_path, self.transform, max_process)

  def __len__(self):
    return len(self.image_infos)

  def __getitem__(self, idx):
    return self.image_infos[idx]['image']

  def transform(self, image, height, width, frame_index, video_file_path):
    self.originals.append(image)
    if width > height:
      ymin = 0
      ymax = height
      xmin = ((width - height) // 2)
      xmax = xmin + height
    else:
      ymin = ((height - width) // 2)
      ymax = ymin + width
      xmin = 0
      xmax = width

    image_cropped = image[ymin:ymax, xmin:xmax]

    return dict(image=cv2.resize(image_cropped, (resize_height, resize_width), interpolation=cv2.INTER_AREA), crop_offset_coord=(ymin, ymax, xmin, xmax), frame_index=frame_index, video_path=video_file_path,
                resize_dim=(resize_height, resize_width))

  def get_face_snapshot(self, all_video_detections, index: int):
    image_info = self.image_infos[index]
    detections_in_frame = all_video_detections[index]

    vid_path = image_info['video_path']
    frame_index = image_info['frame_index']
    o_ymin, o_ymax, o_xmin, o_xmax = image_info['crop_offset_coord']

    # logger.info(f"El 0: {detections_in_frame}: video_path: {str(vid_path)}; frame_index: {frame_index}")

    image_faces_found = []
    for face in detections_in_frame:
      if face.shape[0] == 0:
        image_faces_found.append(dict(frame_index=frame_index, image=None, video_path=vid_path))
        continue

      ymin_frac = face[0]
      xmin_frac = face[1]
      ymax_frac = face[2]
      xmax_frac = face[3]

      image_orig = self.originals[frame_index]
      # logger.info(f"{o_ymin}:{o_ymax}, {o_xmin}:{o_xmax}")

      height_crop = o_ymax - o_ymin
      width_crop = o_xmax - o_xmin

      ymin_oface = int(ymin_frac * height_crop) + o_ymin
      ymax_oface = int(ymax_frac * height_crop) + o_ymin
      xmin_oface = int(xmin_frac * width_crop) + o_xmin
      xmax_oface = int(xmax_frac * width_crop) + o_xmin

      xmax_oface, xmin_oface, ymax_oface, ymin_oface = self.pad_face_crop(xmax_oface, xmin_oface, ymax_oface, ymin_oface)
      image_orig_cropped_face = image_orig[ymin_oface:ymax_oface, xmin_oface:xmax_oface]
      image_faces_found.append(dict(frame_index=frame_index, image=image_orig_cropped_face, video_path=vid_path))

    return image_faces_found

  def pad_face_crop(self, x_max, x_min, y_max, y_min):
    margin_vert = .45
    margin_horiz = .12

    height = y_max - y_min
    y_min = int(y_min - (height * margin_vert))
    y_max = int(y_max + (height * margin_vert))

    width = x_max - x_min
    x_min = int(x_min - (width * margin_horiz))
    x_max = int(x_max + (width * margin_horiz))

    return x_max, x_min, y_max, y_min
