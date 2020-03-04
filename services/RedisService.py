import codecs
import json
import pickle
import zlib

import redis

import config

logger = config.create_logger(__name__)


class RedisService:
  redis_client = None

  def __init__(self):
    port = 6379
    hostname = "localhost"
    password = ""
    self.app_key = "kaggleDeepFakeDetection"

    logger.info(f'About to attempt to start Redis client ...')
    self.redis_client = redis.StrictRedis(
      host=hostname,
      port=port,
      password=password,
      decode_responses=True,
      socket_timeout=100
    )
    logger.info(f'Redis client started.')

  def write_string(self, key: str, value: str):
    self.redis_client.set(key, value)

  def write_as_json(self, key: str, value: object):
    self.redis_client.set(self._compose_key(key), json.dumps(value))

  def _compose_key(self, key):
    return f'{self.app_key}_{key}'

  def read(self, key):
    return self.redis_client.get(self._compose_key(key))

  def read_as_json(self, key):
    result = self.redis_client.get(self._compose_key(key))
    return None if result is None else json.loads(result)

  def write_binary(self, key: str, obj: object):
    dumped_str = zlib.compress(pickle.dumps(obj))
    encoded_str = codecs.encode(dumped_str, "base64").decode()
    self.redis_client.set(self._compose_key(key), encoded_str)

  def read_binary(self, key):
    redis_str = self.redis_client.get(self._compose_key(key))

    obj = None
    if redis_str is not None:
      encoded_str = codecs.decode(redis_str.encode(), "base64")
      pickled = zlib.decompress(encoded_str)
      obj = pickle.loads(pickled)

    return obj

  def close_client_connection(self):
    self.redis_client.close()
