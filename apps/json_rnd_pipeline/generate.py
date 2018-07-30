#!/usr/bin/env python3
"""Parametrized generation of random pipelines."""

import argparse
import os
import copy
import subprocess
import time

from multiprocessing import Process, Queue, Pool, JoinableQueue


class GeneratorParams(object):
  """Parameters driving the generator"""
  def __init__(self, hl_target, hl_seed, pipe_seed, stages,
               dropout, beam, timeout):
    self.hl_target = hl_target
    self.hl_seed = hl_seed
    self.pipeline_seed = pipe_seed
    self.stages = stages
    self.dropout = dropout
    self.beam = beam

    self.timeout = timeout

  def __repr__(self):
    return "GeneratorParams<sched{},pipe{},stages{}>".format(
      self.hl_seed, self.pipeline_seed, self.stages)

  def env(self):
    return {
      "HL_TARGET": self.hl_target,
      "HL_SEED": str(self.hl_seed),
      "PIPELINE_SEED": str(self.pipeline_seed),
      "STAGES": str(self.stages),
      "HL_DROPOUT": str(self.dropout),
      "HL_BEAM_SIZE": str(self.beam),
    }

def get_pipeline_env(params):
  """Preserve user env, and add pipeline-relevant vars."""
  env = copy.deepcopy(os.environ)
  env.update(params.env())
  return env


def build_one(q):
  while True:
    params = q.get(block=True, timeout=None)
    env = get_pipeline_env(params)

    start = time.time()
    subprocess.check_output(["make", "build"], env=env, timeout=params.timeout)
    elapsed = time.time() - start

    q.task_done()

    print("pid {} {}, compiled in {}s".format(os.getpid(), params, elapsed))


def main(args):
  print("Building shared binaries")
  subprocess.run(["make", "build_shared"])

  #TODO(mgharbi): add metadata about the pipe gen
  q = JoinableQueue()
  pool = Pool(1, build_one, (q, ))

  # Distributed build
  print("\nBuilding pipelines")
  for b in range(args.batches):
    for p in range(args.pipelines):
      stages = (p % 30) + 10  # number of stages for that pipeline
      for s in ["root"] + list(range(args.schedules)):
        params = GeneratorParams(
          args.hl_target, s, p, stages, args.dropout,
          args.beam_size, args.timeout)
        q.put(params, block=True)
  q.join()

  # Sequential benchmarking
  print("\nBenchmarking pipelines")
  for b in range(args.batches):
    for p in range(args.pipelines):
      stages = (p % 30) + 10  # number of stages for that pipeline
      for s in ["root"] + list(range(args.schedules)):
        params = GeneratorParams(
          args.hl_target, s, p, stages, args.dropout,
          args.beam_size, args.timeout)
        env = get_pipeline_env(params)
        env["HL_NUM_THREASD"] = str(os.cpu_count())
        subprocess.run(["make", "bench"], env=env, timeout=params.timeout)  # numactl --cpunodebind=0 make bench

  # Gather dataset


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--workers", type=int, default=4)
  parser.add_argument("--batches", type=int, default=1)
  parser.add_argument("--pipelines", type=int, default=1)
  parser.add_argument("--schedules", type=int, default=2)
  parser.add_argument("--hl_target", type=str, default="host-new_autoscheduler")
  parser.add_argument("--dropout", type=int, default=50)
  parser.add_argument("--beam_size", type=int, default=2)
  parser.add_argument("--timeout", type=float, default=10.0, help="in seconds")
  args = parser.parse_args()
  main(args)
