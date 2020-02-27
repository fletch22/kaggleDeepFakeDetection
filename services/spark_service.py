import findspark
from pyspark import SparkContext, MarshalSerializer

import config

logger = config.create_logger(__name__)


def execute(collection, fnProcessInSpark, num_slices=0):
  findspark.init()

  sc = None
  try:
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("INFO")
    print(sc._jsc.sc().uiWebUrl().get())
    rdd = sc.parallelize(collection, numSlices=num_slices)
    results = rdd.map(fnProcessInSpark).collect()
  finally:
    if sc is not None:
      sc.stop()

  return results
