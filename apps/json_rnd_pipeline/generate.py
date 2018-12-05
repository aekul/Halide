#!/usr/bin/env python3
"""Parametrized generation of random pipelines."""

import argparse
from copy import deepcopy
import msgpack
from multiprocessing import Process, Queue, Pool, JoinableQueue, cpu_count
import os
import platform
import re
import subprocess
import time

SEED_RE = re.compile("(.*?_seed)(?P<seed>[^_]*)(_.*?)")


class GeneratorParams(object):
  """Parameters driving the generator, passed as env-vars"""
  def __init__(self, hl_target, hl_seed, pipe_seed, stages,
               dropout, beam, timeout, predictor_url, bin_dir, num_cores,
               llc_size, balance, use_predictor_server):
    self.hl_target = hl_target
    self.hl_seed = hl_seed
    self.pipeline_seed = pipe_seed
    self.stages = stages
    self.dropout = dropout
    self.beam = beam
    self.predictor_url = predictor_url if use_predictor_server else ""
    self.bin_dir = bin_dir
    self.num_cores = num_cores
    self.llc_size = llc_size 
    self.balance = balance

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
      "HL_THROUGHPUT_PREDICTOR_URL": self.predictor_url,
      "BIN": self.bin_dir,
      "HL_MACHINE_PARAMS": "{},{},{}".format(self.num_cores, self.llc_size, self.balance),
    }


def get_machine_info():
  info = {}
  info[b"platform"] = platform.platform().encode()
  info[b"num_cpu"] = cpu_count()
  info[b"name"] = platform.node().encode()
  try:
    ret = subprocess.check_output("git rev-parse HEAD", shell=True)
    info[b"git_commit"] = ret
  except CalledProcessError:
    # unsuccessful call to git
    info[b"git_commit"] = None

  try:
    ret = subprocess.check_output("lscpu", shell=True)
    info[b"cpu_info"] = ret
  except CalledProcessError:
    # unsuccessful call to lscpu
    info[b"cpu_info"] = None

  return info


def get_pipeline_env(params):
  """Preserves user env, and add pipeline-relevant vars."""
  env = deepcopy(os.environ)
  env.update(params.env())
  return env


def build_one(q):
  count = 0
  compile_start = time.time()

  """Build one random pipeline."""
  while True:
    params = q.get(block=True, timeout=None)
    env = get_pipeline_env(params)

    try:
      start = time.time()
      subprocess.check_output(["make", "build"], env=env, timeout=params.timeout)
      elapsed = time.time() - start
      count += 1
      print("    pid {} {}, compiled in {:.2f}s".format(os.getpid(), params, elapsed))
      m, s = divmod(int(time.time() - compile_start), 60)
      h, m = divmod(m, 60)
      if count % 25 == 0:
        print("Compiled {} programs in {:02d}h:{:02d}m:{:02d}s".format(count, h, m, s))
    except subprocess.TimeoutExpired:
      print("    pid {} {}, timed out over {:.2f}s".format(os.getpid(), params, params.timeout))
    except subprocess.CalledProcessError:
      for k in params.env():
        print("{}={} \\".format(k, params.env()[k]))
      print("    pid {} {}, error".format(os.getpid(), params.env()))
    q.task_done()


def get_hl_target(seed="root"):
  if seed == "master":
    return "host"
  else:
    return "host-new_autoscheduler"


def gather_features(schedule_root, pipeline_seed, machine_info):
  features_path = os.path.join(schedule_root, "features.mp")
  with open(features_path, "rb") as fid:
    features = msgpack.load(fid)

  times_path = features_path.replace("features", "timing")
  with open(times_path, "rb") as fid:
    timing = msgpack.load(fid)

  root_path = re.sub(SEED_RE, "\g<1>root\g<3>", schedule_root)
  times_root_path = os.path.join(root_path, "timing.mp")
  with open(times_root_path, "rb") as fid:
    timing_root = msgpack.load(fid)

  features[b"pipeline_seed"] = pipeline_seed
  features[b"time"] = timing[b"time"]
  features[b"time_root"] = timing_root[b"time"]
  features[b"machine_info"] = machine_info
  return features


def main(args):
  results_dir = os.path.abspath(args.results_dir)
  bin_dir = os.path.join("bin")
  # bin_dir = os.path.join(results_dir, "bin")
  curdir = os.path.abspath(os.curdir)
  
  makedir = os.path.dirname(os.path.abspath(__file__))
  print(".Changing directory to Makefile root {}".format(makedir))
  os.chdir(makedir)


  print(".Building shared binaries")
  subprocess.check_output(["make", "build_shared"])

  schedule_seeds = ["root"] + list(range(args.schedules))

  q = JoinableQueue()
  pool = Pool(args.workers, build_one, (q, ))

  num_schedules = len(schedule_seeds)

  machine_info = get_machine_info()

  if not args.gather_only:
    for p in range(args.pipelines):
      pipeline_seed = args.node_id*args.pipelines + p + args.start_seed
      stages = (pipeline_seed % 30) + 2  # number of stages for that pipeline

      print(".Pipeline {} of {} with seed {} | {} stages".format(
        p+1, args.pipelines, pipeline_seed, stages))

      # TODO: check if pipeline has already been generated / benchmarked

      # Distributed build
      print("  .Compile {} schedules".format(num_schedules))
      all_params = []
      for s in schedule_seeds:
        params = GeneratorParams(
          get_hl_target(s), s, pipeline_seed, stages, args.dropout,
          args.beam_size, args.timeout, args.predictor_url, bin_dir,
          args.hl_threads, args.llc_size, args.balance, args.use_predictor_server)
        all_params.append(params)
        q.put(params, block=True)

      # Wait for end of compilation before benchmarking
      q.join()

      # Sequential benchmarking
      print("  .Benchmarking {} schedules".format(num_schedules))
      benchmark_start = time.time()
      for idx_s, s in enumerate(schedule_seeds):
        params = all_params[idx_s]
        env = get_pipeline_env(params)
        env["HL_NUM_THREADS"] = str(args.hl_threads)
        start = time.time()
        try:
          start = time.time()
          subprocess.check_output(["make", "bench"], env=env, timeout=params.timeout)
          elapsed = time.time() - start
          print("    .Benchmarking {} took {:.2f}s".format(params, elapsed))
        except subprocess.CalledProcessError as e:
          for k in env:
            print("{}={} \\".format(k, env[k]))
          print("    .Benchmarking {} errored: {}s".format(params, e))
        except subprocess.TimeoutExpired:
          print("    .Benchmarking {} timed out at {:.2f}s".format(params, params.timeout))

      # Gather training samples to final destination
      data_root = os.path.join(bin_dir, get_hl_target(), "pipe{}_stages{}".format(pipeline_seed, stages))
      print("  .Gathering training samples from {}".format(data_root))
      for r, dd, ff in os.walk(data_root):
        for f in ff:
          if f != "features.mp":
            continue
          try:
            features = gather_features(r, pipeline_seed, machine_info)
            fname = "pipeline_{:03d}_schedule_{:03d}_stages_{}.mp".format(
              pipeline_seed, features[b"schedule_seed"], stages)
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, fname), "wb") as fid:
              msgpack.dump(features, fid)
            print("    ", r)
          except Exception as e:
            print("  Failed {}: {}".format(r, e))

  print(".Saved data to {}".format(results_dir))

  print(".Changing directory back to original location {}".format(curdir))
  os.chdir(curdir)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--results_dir", type=str, required=True)
  parser.add_argument("--workers", type=int, default=cpu_count(), help="number of workers for the parallel build")

  parser.add_argument("--predictor_url", type=str, default="tcp://localhost:5555", help="url of the throughput predictor server, useful when evaluating our predictions")
  parser.add_argument("--use_predictor_server", action="store_true", help="should we use the predictor server?")

  # Selector to run only as subset of steps
  parser.add_argument("--gather_only", dest="gather_only", action="store_true", help="do not generate new pipelines, just consolidate the dataset")

  # Generation params
  parser.add_argument("--pipelines", type=int, default=10000)
  parser.add_argument("--schedules", type=int, default=20)
  parser.add_argument("--dropout", type=int, default=50)
  parser.add_argument("--beam_size", type=int, default=1)
  parser.add_argument("--timeout", type=float, default=20.0, help="in seconds")

  # For distributed generation (to enseure that different nodes generate different
  # groups of programs)
  parser.add_argument("--start_seed", type=int, default=0)
  parser.add_argument("--node_id", type=int, default=0)
  parser.add_argument("--num_nodes", type=int, default=1)

  # Autoscheduler MachineParams
  # Size of last level cache (in bytes); default = 4096KB
  parser.add_argument("--hl_threads", type=int, default=8)
  parser.add_argument("--llc_size", type=int, default=32768)
  parser.add_argument("--balance", type=int, default=40)

  parser.set_defaults(master_scheduler=False)

  parser.set_defaults(gather_only=False)

  args = parser.parse_args()

  main(args)
