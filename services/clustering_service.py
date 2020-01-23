from pathlib import Path
from typing import List
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision.transforms import ToTensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')


def get_embeddings(face_file_paths: List[Path]):
  resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()

  tf_img = lambda i: ToTensor()(i).unsqueeze(0)
  embeddings = lambda input: resnet(input)

  list_embs = []
  with torch.no_grad():
    for face_path in face_file_paths:
      image = Image.open(str(face_path))
      image.resize((160, 160), resample=0)
      t = tf_img(image).to(device)
      e = embeddings(t).squeeze().cpu().tolist()
      list_embs.append(e)

  return list_embs
