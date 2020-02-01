import os
import tempfile
from pathlib import Path
from typing import Any, Tuple
import numpy as np

import librosa
import librosa.display as display
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
import matplotlib.pyplot as plt

import config

logger = config.create_logger(__name__)

def get_audio_clip_from_video(vid_path: Path, start_milli: int, end_milli: int) -> Tuple[Any, Any]:
  video_clip = VideoFileClip(str(vid_path))
  audio_clip: AudioFileClip = video_clip.audio.subclip(start_milli / 1000, end_milli / 1000)

  par_path = tempfile.TemporaryDirectory().name
  os.makedirs(par_path, exist_ok=True)
  output_path = Path(par_path, "foo.wav")
  output_path_str = str(output_path)
  audio_clip.write_audiofile(output_path_str)
  audio_clip.close()

  clip, sample_rate = librosa.load(output_path_str, sr=None)

  return clip, sample_rate



def display_chart(clip, sample_rate):
  n_fft = 1024  # frame length
  hop_length = 512
  stft = librosa.stft(clip, n_fft=n_fft, hop_length=hop_length)
  stft_magnitude, stft_phase = librosa.magphase(stft)
  stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)

  plt.figure(figsize=(12, 6))
  display.specshow(stft_magnitude_db, x_axis='time', y_axis='linear',
                           sr=sample_rate, hop_length=hop_length)

  title = 'n_fft={},  hop_length={},  time_steps={},  fft_bins={}  (2D resulting shape: {})'
  plt.title(title.format(n_fft, hop_length,
                         stft_magnitude_db.shape[1],
                         stft_magnitude_db.shape[0],
                         stft_magnitude_db.shape))
  plt.show()
