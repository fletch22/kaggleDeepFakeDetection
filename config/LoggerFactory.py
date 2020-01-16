import logging
import sys
from logging.handlers import RotatingFileHandler

import config


class LoggerFactory():
  all_loggers = []

  def __init__(self):
    logging_path = config.LOGGING_FILE_PATH
    self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    self.handler_stream = logging.StreamHandler(sys.stdout)
    self.handler_stream.setFormatter(self.formatter)

    self.rot_handler = RotatingFileHandler(logging_path, maxBytes=200000, backupCount=5)
    self.rot_handler.setFormatter(self.formatter)

    self.file_handler = logging.FileHandler(logging_path)

  def create_logger(self, name: str):
    logger = logging.getLogger(name)
    logger.addHandler(self.handler_stream)
    logger.addHandler(self.file_handler)

    logger.setLevel(logging.INFO)
    self.all_loggers.append(logger)

    return logger
