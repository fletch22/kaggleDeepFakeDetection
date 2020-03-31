

from pathlib import Path

import numpy as np

import config
from pipelines.Pipeline import Pipeline
from services import pickle_service

logger = config.create_logger(__name__)


class PredictFromLearner(Pipeline):

  def __init__(self, root_output: Path, pipeline_dirname: str):
    Pipeline.__init__(self, root_output=root_output, pipeline_dirname=pipeline_dirname, overwrite_old_outputs=True, start_output_keys={'output_path'})

  def start(self, **kwargs):
    # pct_train = kwargs['pct_train']
    # pct_test = kwargs['pct_test']



    result = {'output_path': 'foo'}
    self.validate_start_output(result)

    return result
