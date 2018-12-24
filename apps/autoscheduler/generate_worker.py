#!/usr/bin/env python3
"""Parametrized generation of random pipelines."""

import argparse
import msgpack
import subprocess
import time
import pdb
import redis
from rq import Worker, Queue, Connection

def build(pipeline_id, num_schedules, hl_threads):
  print("Building generator")
  subprocess.check_output(["make", "bin/random.generator"])

  print("Building/benchmarking pipeline: {}".format(pipeline_id))
  subprocess.check_output(["bash", "random_pipeline.sh", str(pipeline_id), str(num_schedules), str(hl_threads)])

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--ip", help="server ip", type=str, required=True)
  args = parser.parse_args()

  listen = ["default"]
  conn = redis.from_url("redis://:@{}:6379".format(args.ip))
  with Connection(conn):
    worker = Worker(map(Queue, listen))
    worker.work()

