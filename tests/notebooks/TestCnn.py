import inspect
import sys
import random
from unittest import TestCase

from cv2 import cv2
from fastai.vision import models, Path, cnn_learner, ImageList, error_rate, accuracy, DataBunch, load_data, load_learner, DatasetType
import pandas as pd
from torch.nn.functional import cross_entropy

import config
from pipelines.PredictFromLearner import PredictFromLearner
from services import file_service, batch_data_loader_service, video_service
from util import random_util
from util.BatchData import COL_VID_PATH

logger = config.create_logger(__name__)

class TestCnn(TestCase):

  def test_model_acc(self):
    pred_par_path = Path(config.OUTPUT_MODEL_PAR_PATH, 'preds')
    pred_path = Path(pred_par_path, 'df_test_pred.pkl')

    df_test = pd.read_pickle(pred_path)

    pd.options.display.max_columns = 10
    pd.options.display.max_colwidth = 30
    pd.options.display.max_rows = 999

    df_scored = df_test[['video_name_stem', 'real_or_fake_digit', 'y']]

    # df_scored = df_scored.sort_values(by=['video_name_stem'])
    # logger.info(f'Head: {df_scored.head(100)}')

    # df_grouped = df_scored.groupby('video_name_stem')['y'].mean()
    df_grouped = df_scored.groupby('video_name_stem')

    total = len(df_grouped)

    matched = 0
    for name, group in df_grouped:
      # logger.info(group)
      rof = group.iloc[0,  group.columns.get_loc('real_or_fake_digit')]
      score = group['y'].mean()
      pred = 1 if score > .5 else 0
      if rof == pred:
        matched += 1

      logger.info(f'matched: { group["y"].shape[0]}: {rof == pred}')

    logger.info(f"%Acc: {matched/total}")

    # df_test.sort_by(col)
    logger.info(f'Head: {df_grouped.head(100)}')

  def get_random_swatch(self, image, height, width, frame_index):
    height_new = 224
    width_new = height_new

    max_x = width - width_new
    max_y = height - height_new

    xmin = random.randint(0, max_x)
    ymin = random.randint(0, max_y)

    image_cropped = image[ymin:(ymin + height_new), xmin:(xmin + height_new)]

    # img_resized = cv2.resize(image_cropped, (height_new, width_new), interpolation=cv2.INTER_AREA)

    return image_cropped, height_new, width_new, frame_index


  def test_from_mp4(self):
    # Arrange
    df_batch = batch_data_loader_service.get_all_metadata('c')
    image_par_path = config.OUTPUT_VIRGIN_TEST_IMAGES_PAR_PATH

    # vid_path = df_batch.iloc[0, :][COL_VID_PATH]
    # vid_path = Path('C:\\Kaggle Downloads\\deepfake-detection-challenge\\train\\dfdc_train_part_40\\dkkqtmyitk.mp4')

    data_list = []

    for ndx, row in df_batch.iterrows():
      vid_path = row['vid_path']

      results = video_service.process_specific_video_frames(video_file_path=vid_path, fnProcess=self.get_random_swatch, frames=list(range(0, 25)))
      vid_stem = vid_path.stem
      for r in results:
        image_cropped, height_new, width_new, frame_index = r
        output_path = Path(image_par_path, f'{vid_stem}_{frame_index}.png')
        if output_path.exists():
          output_path.unlink()
        image_converted = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(output_path), image_converted)

        vid_name = vid_path.name
        real_or_fake = df_batch[df_batch['candidate_filename'] == vid_name]['label'].tolist()[0].lower()
        real_or_fake_digit = 1 if real_or_fake == 'fake' else 0
        row = {'vid_path': str(vid_path),
               'filename':output_path.name,
               'path': str(output_path),
               'gross_label': real_or_fake,
               'real_or_fake_digit': real_or_fake_digit,
               'video_name_stem': vid_path.stem}
        data_list.append(row)

    df = pd.DataFrame(data=data_list, columns=['vid_path', 'filename', 'path', 'video_name_stem', 'gross_label', 'real_or_fake_digit'])

    pd.options.display.max_columns = 10
    pd.options.display.max_colwidth = 30
    pd.options.display.max_rows = 999
    logger.info(df.head())

    df_f = df[df['gross_label'] == 'fake']
    df_r = df[df['gross_label'] == 'real']

    num_fakes = df_f.shape[0]
    num_reals = df_r.shape[0]

    logger.info(f'fakes: {num_fakes}; reals: {num_reals}')

    max_num = num_fakes if num_reals > num_fakes else num_reals

    df_bal = pd.concat([df_f.iloc[:max_num,:], df_r.iloc[:max_num,:]])

    logger.info(f'Total: {df_bal.shape[0]}')

    df_path = Path(config.OUTPUT_VIRGIN_TEST_DF_PAR_PATH, 'df_virgin_test.pkl')
    df_bal.to_pickle(df_path)

    # vid_name = f'{df.iloc[0, df.columns.get_loc("video_name_stem")]}.mp4'
    # logger.info(f'vid_name: {vid_name}')
    # matching = df_batch[df_batch['candidate_filename'] == vid_name]['label'].tolist()[0]
    # logger.info(f'Matching: {matching}')

    # logger.info(f'df vid_path: {df.iloc[0, :]["vid_path"]}')

    # logger.info(f'cols: {df_batch.columns}')
    # logger.info(f'lab: {df_batch["label"].tolist()}')
    # logger.info(f'lab: {df_batch["candidate_filename"].tolist()}')
    # logger.info(df_batch.head())

    # get count total videos
    # get equal number fake and real videos
    # get n number sample images from paths
    # output to flat folder with name <video_stem>_<frame>.png

    # image_path = ?
    # df_test = batch_data_loader_service.get_all_metadata('c')
    # df_test['filename'] = df_test[COL_VID_PATH].apply(lambda x: Path(x).name)
    # df_test['video_name_stem'] = df_test[COL_VID_PATH].apply(lambda x: Path(x).stem.split('_')[0])
    # df_test['gross_label'] = df_test['path'].apply(lambda x: 'real' if x.endswith('1.0.png') else 'fake')
    # df_test['real_or_fake_digit'] = df_test['gross_label'].apply(lambda x: 1 if x == 'fake' else 0)
    #
    # logger.info(f'cols: {df_test.columns}')

    # image_path = config.MERGED_SWATCH_IMAGES_PATH
    # pickle_par_path = Path(config.OUTPUT_MODEL_PAR_PATH, 'data')
    # df_test = pd.read_pickle(Path(pickle_par_path, 'df_test'))
    # df_test = df_test.sort_values(by=['video_name_stem'])

    # data = (ImageList.from_df(df_test, image_path, cols='filename'))
    #
    # logger.info(f'data type: {type(data)}')
    #
    # logger.info(f'df_test: {df_test.shape[0]}')
    # logger.info(f'len: {data.items.shape[0]}')
    #
    # learn = load_learner(config.OUTPUT_LEARNER_PAR_PATH, file=config.OUTPUT_LEARNER_CNN_PATH, test=data)
    # learn.model.cuda()
    #
    # preds, _ = learn.get_preds(ds_type=DatasetType.Test)
    #
    # pred_values = preds.data.cpu().numpy()
    #
    # df_test['preds'] = pred_values.tolist()
    #
    # def is_pred_fake(value):
    #   return 0 if value[0] > .5 else 1
    #
    # df_test['y'] = df_test['preds'].apply(is_pred_fake)
    #
    # pred_par_path = Path(config.OUTPUT_MODEL_PAR_PATH, 'preds')
    # pred_par_path.mkdir(exist_ok=True)
    # pred_path = Path(pred_par_path, 'df_test_pred.pkl')
    # df_test.to_pickle(pred_path)

    # Act

    # Assert
    # dir_path = Path(__file__).parent
    # pfl = PredictFromLearner(root_output=dir_path, pipeline_dirname='')
    #
    # pfl.start(images_folder=)
