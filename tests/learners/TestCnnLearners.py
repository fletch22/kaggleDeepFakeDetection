from unittest import TestCase

from fastai.vision import ImageList, Path, LabelList, LabelLists

import config
from learners import cnn_learner

import pandas as pd

logger = config.create_logger(__name__)

class TestCnnLearners(TestCase):

  def test_get(self):
    # Arrange
    df = cnn_learner.get_decorated_df()
    # Act

    # Assert

  def test_ImageList(self):
    # Arrange
    # C:/Kaggle Downloads/deepfake-detection-challenge/output/decorate_df/dataframes/df.pkl
    image_path = Path('C:\\Kaggle Downloads\\deepfake-detection-challenge\\output\\merged\\images')
    path = Path('C:\\Kaggle Downloads\\deepfake-detection-challenge\\output\\decorate_df\\dataframes\\df.pkl')
    df = pd.read_pickle(path)

    df_val = df[df['test_train_split'] == 'validation']
    df_train = df[df['test_train_split'] == 'train']
    df_test = df[df['test_train_split'] == 'test']

    val_path = Path(df_val.iloc[0, df_val.columns.get_loc('path')])
    logger.info(f'Path: {val_path.parent}')

    logger.info(f'DF size: {df_val.shape[0]} rows.')
    logger.info(f'Columns: {df_val.columns}')

    il_train = ImageList.from_df(df_train, image_path, cols='filename').split_none()
    il_val = ImageList.from_df(df_val, image_path, cols='filename').split_none()
    il_test = ImageList.from_df(df_test, image_path, cols='filename').split_none()

    raise Exception("Finish this code.")
    # lls = LabelLists(val_path, il_train, il_val).label_from_lists(df_train['score'].tolist(), df_val['score'].tolist())
    # lls.add_test(il_test, df_test['score'].values)
    #
    print(lls)

  def get_label_list(self, df_val, image_path):
    il_val = ImageList.from_df(df_val, image_path, cols='filename').split_none()
    label_list = il_val.label_from_func(TestCnnLearners.get_label)
    return label_list

  @staticmethod
  def get_label(p):
    path = Path(p)
    score = path.stem.split('_')[2]
    return score