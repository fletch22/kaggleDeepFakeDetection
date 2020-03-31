from unittest import TestCase

from fastai.metrics import error_rate
from fastai.vision import ImageList, Path, LabelList, LabelLists, models, accuracy, cnn_learner, FloatList, create_cnn

import config

import pandas as pd

from learners import cnn_learner as f22_cnn_learner
from services import pickle_service, batch_data_loader_service, file_service
from util.BatchData import BatchData

logger = config.create_logger(__name__)

class TestCnnLearners(TestCase):

  def test_get(self):
    # Arrange
    df = f22_cnn_learner.get_decorated_df()
    # Act

    # Assert

  def test_ImageList(self):
    # Arrange
    # C:/Kaggle Downloads/deepfake-detection-challenge/output/decorate_df/dataframes/df.pkl
    image_path = Path('C:\\Kaggle Downloads\\deepfake-detection-challenge\\output\\merged\\images')
    path = Path('C:\\Kaggle Downloads\\deepfake-detection-challenge\\output\\decorate_df\\dataframes\\df.pkl')
    df = pd.read_pickle(path)

    logger.info(f'DF: {df.head()}')

    df_train = df[df['test_train_split'] == 'train']
    df_val = df[df['test_train_split'] == 'validation']
    df_test = df[df['test_train_split'] == 'test']

    num_fake = df_train[df_train["gross_label"] == "fake"].shape[0]
    num_real = df_train[df_train["gross_label"] == "real"].shape[0]
    logger.info(f'rat: {num_fake}: {num_real}: ')

    return

    #
    #
    # logger.info(f'df_val Index: {type(df_val.index)}')
    #
    # val_path = Path(df_val.iloc[0, df_val.columns.get_loc('path')])
    # logger.info(f'Path: {val_path.parent}')
    #
    # data = (ImageList.from_df(df, image_path, cols='filename')
    #   .split_by_idxs(train_idx=df_train.index, valid_idx=df_val.index)
    #   .label_from_df(cols='score', label_cls=FloatList)
    #   .databunch(bs=32))
    #
    # learn = create_cnn(data, models.resnet34, metrics=[error_rate, accuracy])
    # learn.model.cuda()
    #
    # learn.save('before-learner')
    # learn.lr_find()
    # learn.recorder.plot()

  def test_columns(self):
    df, _ = pickle_service.concat_pickled_dataframes(config.MERGED_SWATCH_DATA_PATH)

    logger.info(f'Cols: {df.columns}')

  def get_label_list(self, df_val, image_path):
    il_val = ImageList.from_df(df_val, image_path, cols='filename').split_none()
    label_list = il_val.label_from_func(TestCnnLearners.get_label)
    return label_list

  @staticmethod
  def get_label(p):
    path = Path(p)
    score = path.stem.split('_')[2]
    return score