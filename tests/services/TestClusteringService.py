from pathlib import Path
from unittest import TestCase

import config
from services import clustering_service, file_service


class TestClusteringService(TestCase):

  def test_embeddings(self):
    # Arrange
    face_paths = [Path(f) for f in file_service.walk(config.SMALL_HEAD_OUTPUT_PATH)]

    # Act
    clustering_service.get_embeddings(face_paths)

    # Assert

  def test_get_embeddings_df(self):
    # Arrange
    # face_paths = [Path(f) for f in file_service.walk(config.SMALL_HEAD_IMAGE_PATH)]
    # D:\Kaggle Downloads\deepfake-detection-challenge\output\small_heads\aaqaifqrwn
    face_paths = file_service.walk("D:\\Kaggle Downloads\\deepfake-detection-challenge\\output\\small_heads\\uakmltagvm")
    # face_paths = file_service.walk("D:\\Kaggle Downloads\\deepfake-detection-challenge\\output\\trash\\owxbbpjpch")

    # Act
    df = clustering_service.get_embeddings_in_df(face_paths)
    # Assert