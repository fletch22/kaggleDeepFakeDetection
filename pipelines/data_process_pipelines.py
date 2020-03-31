import random
from pathlib import Path
from random import shuffle

import pandas as pd
from cv2 import cv2
from fastai.basic_data import load_data
from fastai.vision import ImageList, cnn_learner, models, error_rate, accuracy, load_learner, DatasetType
from sklearn.metrics import log_loss
from torch.nn.functional import cross_entropy

import config
from diff import find_diff_random, find_real_swatches
from diff.DiffSink import DiffSink
from diff.FaceFinder import FaceFinder
from pipelines.DecorateDf import DecorateDf
from services import batch_data_loader_service, video_service
from services.RedisService import RedisService
from util import blazeface_detection, merge_swatches
from util.BatchData import BatchData
from util.FaceDetection import FaceDetection
from util.FaceDetectionIterator import FaceDetectionIterator
import numpy as np

logger = config.create_logger(__name__)


# Detect face locations in all frames.
def pipeline_stage_1():
  output_parent_path = config.FACE_DET_PAR_PATH
  parent_folder_paths = [config.TRAIN_PARENT_PATH_C, config.TRAIN_PARENT_PATH_D]
  video_paths = FaceDetection.get_video_paths(parent_folder_paths=parent_folder_paths, output_parent_path=output_parent_path)

  blazeface_detection.multi_batch_detection(video_paths=video_paths, output_parent_path=output_parent_path)

  df_iterator: FaceDetectionIterator = FaceDetection.get_dataframes_iterator(output_parent_path=output_parent_path)

  for df in df_iterator:
    logger.info(f"Number of rows from first pickle file: {df.shape[0]}")
    break


# Copy face locations to Redis
def pipeline_stage_2():
  fdi = FaceDetectionIterator(config.FACE_DET_PAR_PATH)

  redis_service = RedisService()

  for df_det in fdi:
    for _, row in df_det.iterrows():
      det = row['detections']
      vid_path = Path(row['path'])

      face_finder = FaceFinder(vid_path)
      face_finder.add_detections(det)

      logger.info(f'Writing to redis \'{vid_path.name}\'.')
      redis_service.write_binary(FaceFinder.get_redis_key(vid_path), face_finder)


# Create swatches
def pipeline_stage_3():
  # Arrange
  output_par_path = Path(config.SSIM_RND_DIFFS_OUTPUT_PATH)

  files = batch_data_loader_service.get_metadata_json_files()
  shuffle(files)

  # batch_data = batch_data_loader_service.load_batch(3)
  max_process = 300000
  max_process_per_video = 30
  diff_sink = DiffSink(output_par_path)

  num_processed = 0
  for f in files:
    if num_processed > max_process:
      break
    batch_data: BatchData = batch_data_loader_service.load_batch_from_path(f)
    num_processed += find_diff_random.process_all_diffs(batch_data, diff_sink, output_par_path, max_total_process=max_process, max_process_per_video=max_process_per_video)

  logger.info(f'Total Processed: {num_processed}.')


# Save all metadata to single file
def pipeline_stage_4():
  batch_data_loader_service.save_all_metadata_to_single_df()


def pipeline_stage_5():
  erase_history = False
  max_proc_swatches = 745860

  find_real_swatches.get_real(erase_history=erase_history, max_proc_swatches=max_proc_swatches)


# Copy all image to the same folder'
def pipeline_stage_6():
  destination = config.MERGED_SWATCH_PAR_PATH
  # file_service.delete_files(destination)

  max_process = 2000000
  map = merge_swatches.merge_real_and_fake(destination, max_process, overwrite_existing=True)


def pipeline_stage_7():
  # learn
  root_output = config.OUTPUT_PATH_C
  pipeline_dirname = 'decorate_df'
  pipeline = DecorateDf(root_output=root_output, pipeline_dirname=pipeline_dirname)

  result = pipeline.start(pct_train=80, pct_test=5)

  logger.info(f'Result: {result}')


def pipeline_stage_8():
  pass
  # See cnn.ipynb


def pipeline_stage_9():
  image_path = Path('C:\\Kaggle Downloads\\deepfake-detection-challenge\\output\\merged\\images')
  file_path = Path(config.OUTPUT_MODEL_PAR_PATH, 'cnn_resnet34__2020-03-29T23-56-43-794897.pkl_16')

  pickle_par_path = Path(config.OUTPUT_MODEL_PAR_PATH, 'data')

  # df_test = pd.read_pickle(Path(pickle_par_path, 'df_test'))

  databunch = load_data(bs=32, path=pickle_par_path)

  # data_test = (ImageList.from_df(df=df_test, path=image_path, cols='filename').split_none()
  #              .label_from_df(cols='real_or_fake_digit'))
  # databunch.add_test(data_test, label=None)

  learn = cnn_learner(databunch, models.resnet18, metrics=[error_rate, accuracy, cross_entropy])
  learn.load(file_path)
  learn.model.cuda()

  learn.export(config.OUTPUT_LEARNER_CNN_PATH)


def pipeline_stage_10():
  image_path = config.MERGED_SWATCH_IMAGES_PATH

  pickle_par_path = Path(config.OUTPUT_MODEL_PAR_PATH, 'data')

  df_test = pd.read_pickle(Path(pickle_par_path, 'df_test'))
  df_test = df_test.sort_values(by=['video_name_stem'])

  data = (ImageList.from_df(df_test, image_path, cols='filename'))

  logger.info(f'data type: {type(data)}')

  logger.info(f'df_test: {df_test.shape[0]}')
  logger.info(f'len: {data.items.shape[0]}')

  learn = load_learner(config.OUTPUT_LEARNER_PAR_PATH, file=config.OUTPUT_LEARNER_CNN_PATH, test=data)
  learn.model.cuda()

  preds, _ = learn.get_preds(ds_type=DatasetType.Test)

  pred_values = preds.data.cpu().numpy()

  df_test['preds'] = pred_values.tolist()

  def is_pred_fake(value):
    return 0 if value[0] > .5 else 1

  df_test['y'] = df_test['preds'].apply(is_pred_fake)

  pred_par_path = Path(config.OUTPUT_MODEL_PAR_PATH, 'preds')
  pred_par_path.mkdir(exist_ok=True)
  pred_path = Path(pred_par_path, 'df_test_pred.pkl')
  df_test.to_pickle(pred_path)


def pipeline_stage_11():
  pred_par_path = Path(config.OUTPUT_MODEL_PAR_PATH, 'preds')
  pred_path = Path(pred_par_path, 'df_test_pred.pkl')

  df_test = pd.read_pickle(pred_path)

  pd.options.display.max_columns = 10
  pd.options.display.max_colwidth = 30
  pd.options.display.max_rows = 999

  logger.info(f'Columns: {df_test.columns}')

  df_scored = df_test[['video_name_stem', 'real_or_fake_digit', 'y', 'pred_raw']]
  df_grouped = df_scored.groupby('video_name_stem')

  total = len(df_grouped)

  matched = 0
  y = []
  y_pred = []
  y_pred_raw = []
  for name, group in df_grouped:
    # logger.info(group)
    rof = group.iloc[0, group.columns.get_loc('real_or_fake_digit')]

    logger.info(f'rof type {type(rof)}')

    score = group['y'].mean()
    score_raw = group['pred_raw'].mean()

    pred = 1 if score > .5 else 0

    y_pred_raw.append(score_raw)
    y_pred.append(pred)
    y.append(rof.item())

    if rof == pred:
      matched += 1

    logger.info(f'matched: {group["y"].shape[0]}: {rof == pred}')

  logger.info(f"%Acc: {matched / total}")


  logger.info(f'y: {y}')
  logger.info(f'y: {y_pred}')
  logger.info(f'y: {y_pred_raw}')

  loss = log_loss(y, y_pred_raw)

  logger.info(f"LogLoss: {loss}")

  # def cross_entropy(predictions, targets):
  #   N = predictions.shape[0]
  #   ce = -np.sum(targets * np.log(predictions)) / N
  #   return ce
  #
  # ce = cross_entropy(np.array(y_pred), np.array([y]))
  # logger.info(f"ce: {ce}")



def pipeline_stage_12():
  df_batch = batch_data_loader_service.get_all_metadata('c')
  image_par_path = config.OUTPUT_VIRGIN_TEST_IMAGES_PAR_PATH

  df_batch = df_batch.sample(frac=1).reset_index(drop=True)

  def get_random_swatch(image, height, width, frame_index):
    height_new = 224
    width_new = height_new

    max_x = width - width_new
    max_y = height - height_new

    xmin = random.randint(0, max_x)
    ymin = random.randint(0, max_y)

    image_cropped = image[ymin:(ymin + height_new), xmin:(xmin + height_new)]

    return image_cropped, height_new, width_new, frame_index

  total_needed = 4000
  df_f = df_batch[df_batch['label'] == 'FAKE'].iloc[:(total_needed//2),:]
  df_r = df_batch[df_batch['label'] == 'REAL'].iloc[:(total_needed//2),:]
  df_bal = pd.concat([df_f, df_r])

  logger.info(f'Total rows to process: {df_bal.shape[0]}')

  data_list = []
  num_vid_frames = 50
  for ndx, row in df_bal.iterrows():
    logger.info(f"Video: {ndx}/{df_bal.shape[0]}")
    vid_path = row['vid_path']
    vid_stem = vid_path.stem

    results = video_service.process_specific_video_frames(video_file_path=vid_path, fnProcess=get_random_swatch, frames=list(range(0, num_vid_frames)))

    for r in results:
      image_cropped, height_new, width_new, frame_index = r
      output_path = Path(image_par_path, f'{vid_stem}_{frame_index}.png')

      if not output_path.exists():
        image_converted = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(output_path), image_converted)

      vid_name = vid_path.name
      real_or_fake = df_bal[df_bal['candidate_filename'] == vid_name]['label'].tolist()[0].lower()
      real_or_fake_digit = 1 if real_or_fake == 'fake' else 0
      row = {'vid_path': str(vid_path),
             'filename': output_path.name,
             'path': str(output_path),
             'gross_label': real_or_fake,
             'real_or_fake_digit': real_or_fake_digit,
             'video_name_stem': vid_path.stem}
      data_list.append(row)

  df = pd.DataFrame(data=data_list, columns=['vid_path', 'filename', 'path', 'video_name_stem', 'gross_label', 'real_or_fake_digit'])

  # df_path = file_service.get_unique_persist_filename(parent_path=config.OUTPUT_VIRGIN_TEST_DF_PAR_PATH,
  #                                                    base_output_stem='df_virgin_test',
  #                                                    extension='pkl',
  #                                                    use_date=True)
  df_path = Path(config.OUTPUT_VIRGIN_TEST_DF_PAR_PATH, 'df_virgin_test.pkl')

  df.to_pickle(df_path)


def pipeline_stage_13():
  image_path = config.OUTPUT_VIRGIN_TEST_IMAGES_PAR_PATH

  df_path = Path(config.OUTPUT_VIRGIN_TEST_DF_PAR_PATH, 'df_virgin_test.pkl')
  df_test = pd.read_pickle(df_path)
  df_test = df_test.sort_values(by=['video_name_stem'])

  data = (ImageList.from_df(df_test, image_path, cols='filename'))

  logger.info(f'df_test num rows: {df_test.shape[0]}')

  learn = load_learner(config.OUTPUT_LEARNER_PAR_PATH, file=config.OUTPUT_LEARNER_CNN_PATH, test=data)
  learn.model.cuda()

  preds, _ = learn.get_preds(ds_type=DatasetType.Test)

  pred_values = preds.data.cpu().numpy()

  logger.info(f'pred_values: {pred_values[:10]}')

  df_test['preds'] = pred_values.tolist()

  def get_pred(x):
    return x[1]

  df_test['pred_raw'] = df_test['preds'].apply(get_pred)

  logger.info(f'df_test cols: {df_test.columns}')

  def is_pred_fake(value):
    return 0 if value[0] > .5 else 1

  df_test['y'] = df_test['preds'].apply(is_pred_fake)

  pred_par_path = Path(config.OUTPUT_MODEL_PAR_PATH, 'preds')
  pred_par_path.mkdir(exist_ok=True)
  pred_path = Path(pred_par_path, 'df_test_pred.pkl')
  df_test.to_pickle(pred_path)

def pipeline_stage_14():
  pipeline_stage_11()

if __name__ == '__main__':
  pipeline_stage_14()

