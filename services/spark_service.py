import findspark
from pyspark import SparkContext

import config

logger = config.create_logger(__name__)


def execute(collection, fnProcessInSpark):
  findspark.init()

  sc = None
  try:
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("INFO")
    print(sc._jsc.sc().uiWebUrl().get())
    rdd = sc.parallelize(collection, numSlices=None)
    results = rdd.map(fnProcessInSpark).collect()
  finally:
    if sc is not None:
      sc.stop()

  return results
