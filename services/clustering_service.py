from pathlib import Path
from typing import List

import PIL
import pandas as pd
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision.transforms import ToTensor

import config
from services import image_service
import numpy as np

logger = config.create_logger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')

def get_embeddings_in_df(face_file_paths: List[Path]):
  list_embs = get_embeddings(face_file_paths)
  df = pd.DataFrame({'face': face_file_paths[:len(list_embs)], 'embedding': list_embs})
  df['video'] = df.face.apply(lambda x: f'{Path(x).stem}.mp4')

  def get_batch_number(video_value):
    logger.info(video_value)

  df['chunk'] = df.video.apply(lambda x: get_batch_number(x))
  df = df[['video', 'face', 'chunk', 'embedding']]

  return df

def get_embeddings(face_file_paths: List[Path]):
  resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()

  tf_img = lambda i: ToTensor()(i).unsqueeze(0)
  embeddings = lambda input: resnet(input)

  list_embs = []
  with torch.no_grad():
    for ndx, face_path in enumerate(face_file_paths):
      pil_image = Image.open(str(face_path))
      width, height = pil_image.size

      if width < 160 or height < 160:
        pil_image = pil_image.resize((160, 160), resample=PIL.Image.NEAREST)

      # open_cv_image = np.array(pil_image)
      # image_service.show_image(open_cv_image)
      
      t = tf_img(pil_image).to(device)
      e = embeddings(t).squeeze().cpu().tolist()
      list_embs.append(e)

  return list_embs
