#!/usr/bin/env python3
"""Parametrized generation of random pipelines."""

import argparse
from copy import deepcopy
import json
from multiprocessing import Process, Queue, Pool, JoinableQueue
import os
import re
import subprocess
import time


class GeneratorParams(object):
  """Parameters driving the generator, passed as env-vars"""
  def __init__(self, hl_target, hl_seed, pipe_seed, stages,
               dropout, beam, timeout, predictor_url):
    self.hl_target = hl_target
    self.hl_seed = hl_seed
    self.pipeline_seed = pipe_seed
    self.stages = stages
    self.dropout = dropout
    self.beam = beam
    self.predictor_url = predictor_url

    self.timeout = timeout

  def __repr__(self):
    return "GeneratorParams<sched{},pipe{},stages{}>".format(
      self.hl_seed, self.pipeline_seed, self.stages)

  def env(self):
    return {
      "HL_TARGET": self.hl_target,
      "HL_SEED": str(self.hl_seed),
      "PIPELINE_SEED": str(self.pipeline_seed),
      "PIPELINE_STAGES": str(self.stages),
      "HL_RANDOM_DROPOUT": str(self.dropout),
      "HL_BEAM_SIZE": str(self.beam),
      #"HL_THROUGHPUT_PREDICTOR_URL": self.predictor_url,
    }

def get_pipeline_env(params):
  """Preserves user env, and add pipeline-relevant vars."""
  env = deepcopy(os.environ)
  env.update(params.env())
  return env


def build_one(q):
  """Build one random pipeline."""
  while True:
    params = q.get(block=True, timeout=None)
    env = get_pipeline_env(params)

    try:
      start = time.time()
      subprocess.check_output(["make", "build"], env=env, timeout=params.timeout)
      elapsed = time.time() - start
      print("pid {} {}, compiled in {:.2f}s".format(os.getpid(), params, elapsed))
    except subprocess.TimeoutExpired:
      print("pid {} {}, timed out over {:.2f}s".format(os.getpid(), params, params.timeout))

    q.task_done()


def get_hl_target(seed="root"):
  if seed == "master":
    return "host"
  else:
    return "host-new_autoscheduler"

def main(args):
  print("Building shared binaries")
  subprocess.check_output(["make", "build_shared"])

  if args.evaluate:
    schedule_seeds = ["root", "master", 0]
  else:
    schedule_seeds = ["root"] + list(range(args.schedules))

  #TODO(mgharbi): add metadata about the pipe gen

  q = JoinableQueue()
  pool = Pool(args.workers, build_one, (q, ))

  if not args.gather_only:
    # Distributed build
    if not args.bench_only:
      print("\nBuilding pipelines")
      for p in range(args.pipelines):
        stages = (p % 30) + 10  # number of stages for that pipeline
        for s in schedule_seeds:
          params = GeneratorParams(
            get_hl_target(s), s, p, stages, args.dropout,
            args.beam_size, args.timeout, args.predictor_url)
          q.put(params, block=True)
      q.join()

      if args.build_only:
        return

    # Sequential benchmarking
    print("\nBenchmarking pipelines")
    for p in range(args.pipelines):
      stages = (p % 30) + 10  # number of stages for that pipeline
      for s in schedule_seeds:
        params = GeneratorParams(
          get_hl_target(s), s, p, stages, args.dropout,
          args.beam_size, args.timeout, args.predictor_url)
        env = get_pipeline_env(params)
        env["HL_NUM_THREASD"] = str(os.cpu_count())
        start = time.time()
        try:
          start = time.time()
          subprocess.check_output(["make", "bench"], env=env, timeout=params.timeout)
          # numactl --cpunodebind=0 make bench
          elapsed = time.time() - start
          print("Benchmarking {} took {:.2f}s".format(params, elapsed))
        except subprocess.CalledProcessError as e:
          print("Benchmarking {} errored: {}s".format(params, e))
        except subprocess.TimeoutExpired:
          print("Benchmarking {} timed out at {:.2f}s".format(params, params.timeout))

    if args.build_only:
      return

  src = os.path.join(args.bin_dir, get_hl_target())
  os.makedirs(args.results_dir, exist_ok=True)
  path_re = re.compile(r".*pipe(?P<pipe>\d+)")
  seed_re = re.compile("(.*?_seed)(?P<seed>[^_]*)(_.*?)")
  if args.evaluate:
    print("\nConsolidating timing reports")
    pipe_seeds = []
    root_times = []
    master_times = []
    new_times = []
    for r, dd, ff in os.walk(src):
      for f in ff:
        if f == "features.json":
          f = "timing.json"
          match = path_re.match(r)
          pipe_seed = int(match.group("pipe"))
          root_path = re.sub(seed_re, "\g<1>root\g<3>", r)
          master_path = re.sub(seed_re, "\g<1>master\g<3>", r).replace(
            get_hl_target(), get_hl_target("master"))

          with open(os.path.join(r, f), 'r') as fid:
            new_time = json.load(fid)["time"]

          with open(os.path.join(root_path, f), 'r') as fid:
            root_time = json.load(fid)["time"]

          with open(os.path.join(master_path, f), 'r') as fid:
            master_time = json.load(fid)["time"]

          pipe_seeds.append(pipe_seed)
          root_times.append(root_time)
          master_times.append(master_time)
          new_times.append(new_time)
          print(pipe_seed, root_time, master_time, new_time)

    with open(os.path.join(args.results_dir, "evaluation_report.json"), 'w') as fid:
      report = {
        "pipeline": pipe_seeds,
        "root_time": root_times,
        "master_time": master_times,
        "new_time": new_times,
      }
      json.dump(report, fid)

  else:
    # Gather training dataset
    print("\nGathering training dataset")
    for r, dd, ff in os.walk(src):
      for f in ff:
        if f == "features.json":
          start = time.time()
          # extract pipe seed 
          match = path_re.match(r)
          pipe_seed = int(match.group("pipe"))

          root_path = re.sub(seed_re, "\g<1>root\g<3>", r)

          try:
            feats = os.path.join(r, f)
            with open(feats, 'r') as fid:
              features = json.load(fid)
            times = feats.replace("features", "timing")
            with open(times, 'r') as fid:
              timing = json.load(fid)
            times_root = os.path.join(root_path, "timing.json")
            with open(times_root, 'r') as fid:
              timing_root = json.load(fid)

            features["pipeline_seed"] = pipe_seed
            features["time"] = timing["time"]
            features["time_root"] = timing_root["time"]

            elapsed = time.time() - start
            print(r, elapsed, pipe_seed, features["schedule_seed"], features["time"], features["time_root"])

            fname = "pipeline_{:03d}_schedule_{:03d}.json".format(pipe_seed, features["schedule_seed"])
            with open(os.path.join(args.results_dir, fname), 'w') as fid:
              json.dump(features, fid)
          except:
            print(r, "failed")



if __name__ == "__main__":
  # TODO: add mechanism to launch
  parser = argparse.ArgumentParser()
  parser.add_argument("--workers", type=int, default=4)
  parser.add_argument("--results_dir", type=str, default="generated")
  parser.add_argument("--bin_dir", type=str, default="bin")

  parser.add_argument("--evaluate", dest="evaluate", action="store_true", help="evaluate autoscheduler, instead of generating data samples")
  parser.add_argument("--predictor_url", type=str, default="tcp://localhost:5555", help="url of the throughput predictor server, useful when evaluating our predictions")

  # Selector to run only as subset of steps
  parser.add_argument("--build_only", dest="build_only", action="store_true", help="do not benchmark the pipelines")
  parser.add_argument("--bench_only", dest="bench_only", action="store_true", help="do not generate new pipelines, just consolidate the dataset")
  parser.add_argument("--gather_only", dest="gather_only", action="store_true", help="do not generate new pipelines, just consolidate the dataset")

  # Generation params
  parser.add_argument("--pipelines", type=int, default=10000)
  parser.add_argument("--schedules", type=int, default=20)
  parser.add_argument("--dropout", type=int, default=50)
  parser.add_argument("--beam_size", type=int, default=1)
  parser.add_argument("--timeout", type=float, default=20.0, help="in seconds")

  parser.set_defaults(master_scheduler=False)

  parser.set_defaults(build_only=False)
  parser.set_defaults(bench_only=False)
  parser.set_defaults(gather_only=False)

  args = parser.parse_args()

  main(args)
