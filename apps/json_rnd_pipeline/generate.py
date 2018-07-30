#!/usr/bin/env python3
"""Parametrized generation of random pipelines."""

import argparse
import os
import subprocess
from multiprocessing import Process, Queue, Pool, JoinableQueue

class GeneratorParams(object):
  """Parameters driving the generator"""
  def __init__(self, hl_target, hl_seed, pipe_seed, stages,
               dropout, beam):
    self.hl_target = hl_target
    self.hl_seed = hl_seed
    self.pipeline_seed = pipe_seed
    self.stages = stages
    self.dropout = dropout
    self.beam = beam

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


def build_one(q):
  while True:
    params = q.get(block=True, timeout=None)
    print("Process", os.getpid(), params)
    subprocess.run(["make", "build"], env=params.env())
    # TODO: timeout on compile
    # ret = subprocess.run(["./test.sh"], env=params.env())
    q.task_done()

def run_one(q):
  pass

def main(args):
  # print("Building shared binaries")
  subprocess.run(["make", "build"])

  #TODO(mgharbi): add metadata about the pipe gen
  q = JoinableQueue()
  pool = Pool(1, build_one, (q, ))

  # Distributed build
  for b in range(args.batches):
    for p in range(args.pipelines):
      # number of stages for that pipeline
      stages = (p % 30) + 10
      params = GeneratorParams(
        args.hl_target, "root", p, stages, args.dropout, args.beam_size)
      q.put(params, block=True)
      for s in range(args.schedules):
        params = GeneratorParams(
          args.hl_target, s, p, stages, args.dropout, args.beam_size)
        q.put(params, block=True)
  print("Waiting for queue completion")
  q.join()

  # Sequential benchmarking

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--workers", type=int, default=4)
  parser.add_argument("--batches", type=int, default=1)
  parser.add_argument("--pipelines", type=int, default=1)
  parser.add_argument("--schedules", type=int, default=2)
  parser.add_argument("--hl_target", type=str, default="host-new_autoscheduler")
  parser.add_argument("--dropout", type=int, default=50)
  parser.add_argument("--beam_size", type=int, default=2)
  args = parser.parse_args()
  main(args)
