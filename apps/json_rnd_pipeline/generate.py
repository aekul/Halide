#!/usr/bin/env python3
"""Parametrized generation of random pipelines."""
#TODO(mgharbi): add metadata about the pipe gen

import argparse
from copy import deepcopy
import msgpack
from multiprocessing import Process, Queue, Pool, JoinableQueue, cpu_count
import os
import platform
import re
import subprocess
import time


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
      print("pid {} {}, compiled in {:.2f}s".format(os.getpid(), params, elapsed))
      m, s = divmod(int(time.time() - compile_start), 60)
      h, m = divmod(m, 60)
      if count % 25 == 0:
        print("Compiled {} programs in {:02d}h:{:02d}m:{:02d}s".format(count, h, m, s))
    except subprocess.TimeoutExpired:
      print("pid {} {}, timed out over {:.2f}s".format(os.getpid(), params, params.timeout))
    except subprocess.CalledProcessError:
      for k in params.env():
        print("{}={} \\".format(k, params.env()[k]))
      print("pid {} {}, error".format(os.getpid(), params.env()))

    q.task_done()

  print("Finished build_one(): compiled {} programs".format(count))


def get_hl_target(seed="root"):
  if seed == "master":
    return "host"
  else:
    return "host-new_autoscheduler"


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

  if args.evaluate:
    schedule_seeds = ["root", "master", 10000]
  else:
    schedule_seeds = ["root"] + list(range(args.schedules))

  q = JoinableQueue()
  pool = Pool(args.workers, build_one, (q, ))

  if not args.gather_only:
    # Distributed build
    if not args.bench_only:
      print(".Building pipelines")
      for p in range(args.pipelines):
        pipeline_seed = args.node_id + (p * args.num_nodes)
        stages = (p % 30) + 2  # number of stages for that pipeline
        for s in schedule_seeds:
          params = GeneratorParams(
            get_hl_target(s), s, pipeline_seed, stages, args.dropout,
            args.beam_size, args.timeout, args.predictor_url, bin_dir,
            args.num_cores, args.llc_size, args.balance, args.use_predictor_server)
          q.put(params, block=True)
      q.join()

      if args.build_only:
        return

    # Sequential benchmarking
    total = args.pipelines * len(schedule_seeds)
    print(".Benchmarking pipelines: {} total".format(total))
    completed = 0
    benchmark_start = time.time()
    for p in range(args.pipelines):
      pipeline_seed = args.node_id + (p * args.num_nodes)
      stages = (p % 30) + 2  # number of stages for that pipeline
      for s in schedule_seeds:
        params = GeneratorParams(
          get_hl_target(s), s, pipeline_seed, stages, args.dropout,
          args.beam_size, args.timeout, args.predictor_url, bin_dir,
          args.num_cores, args.llc_size, args.balance, args.use_predictor_server)
        env = get_pipeline_env(params)
        env["HL_NUM_THREADS"] = str(args.hl_threads)
        start = time.time()
        try:
          start = time.time()
          subprocess.check_output(["make", "bench"], env=env, timeout=params.timeout)
          # numactl --cpunodebind=0 make bench
          elapsed = time.time() - start
          print("  .Benchmarking {} took {:.2f}s".format(params, elapsed))
          completed += 1
        except subprocess.CalledProcessError as e:
          for k in env:
            print("{}={} \\".format(k, env[k]))
          print("  .Benchmarking {} errored: {}s".format(params, e))
        except subprocess.TimeoutExpired:
          print("  .Benchmarking {} timed out at {:.2f}s".format(params, params.timeout))

        if completed % 100 == 0:
          m, s = divmod(int(time.time() - benchmark_start), 60)
          h, m = divmod(m, 60)
          print(".Benchmarked {} / {} in {:02d}h:{:02d}m:{:02d}s".format(completed, total, h, m, s))

    if args.build_only:
      return

  src = os.path.join(bin_dir, get_hl_target())
  os.makedirs(results_dir, exist_ok=True)
  path_re = re.compile(r".*pipe(?P<pipe>\d+)")
  seed_re = re.compile("(.*?_seed)(?P<seed>[^_]*)(_.*?)")
  stage_re = re.compile(".*stages(\d+)")
  if args.evaluate:
    print("\nConsolidating timing reports")
    pipe_seeds = []
    root_times = []
    master_times = []
    new_times = []
    for r, dd, ff in os.walk(src):
      for f in ff:
        if f == "features.msgpack":
          f = "timing.mp"
          match = path_re.match(r)
          pipe_seed = int(match.group("pipe"))
          root_path = re.sub(seed_re, "\g<1>root\g<3>", r)
          master_path = re.sub(seed_re, "\g<1>master\g<3>", r).replace(
            get_hl_target(), get_hl_target("master"))

          try:
            with open(os.path.join(r, f), "r") as fid:
              new_time = msgpack.load(fid)["time"]

            with open(os.path.join(root_path, f), "r") as fid:
              root_time = msgpack.load(fid)["time"]

            with open(os.path.join(master_path, f), "r") as fid:
              master_time = msgpack.load(fid)["time"]

            pipe_seeds.append(pipe_seed)
            root_times.append(root_time)
            master_times.append(master_time)
            new_times.append(new_time)
            print(pipe_seed, root_time, master_time, new_time)
          except:
            pass

    with open(os.path.join(results_dir, "evaluation_report.mp"), "w") as fid:
      report = {
        "pipeline": pipe_seeds,
        "root_time": root_times,
        "master_time": master_times,
        "new_time": new_times,
      }
      msgpack.dump(report, fid)

  else:
    machine_info = get_machine_info()
    # Gather training dataset
    print("\nGathering training dataset")
    completed = 0
    gather_start = time.time()
    for r, dd, ff in os.walk(src):
      for f in ff:
        if f == "features.mp":
          start = time.time()
          # extract pipe seed 
          match = path_re.match(r)
          pipe_seed = int(match.group("pipe"))

          root_path = re.sub(seed_re, "\g<1>root\g<3>", r)
          num_stages = int(stage_re.match(r).group(1))

          try:
            feats = os.path.join(r, f)
            with open(feats, "rb") as fid:
              features = msgpack.load(fid)
            times = feats.replace("features", "timing")
            with open(times, "rb") as fid:
              timing = msgpack.load(fid)

            times_root = os.path.join(root_path, "timing.mp")
            with open(times_root, "rb") as fid:
              timing_root = msgpack.load(fid)

            features[b"pipeline_seed"] = pipe_seed
            features[b"time"] = timing[b"time"]
            features[b"time_root"] = timing_root[b"time"]

            features[b"machine_info"] = machine_info

            elapsed = time.time() - start
            if (completed - 1) % 1000 == 0:
              m, s = divmod(int(time.time() - gather_start), 60)
              h, m = divmod(m, 60)
              print(r, elapsed, pipe_seed, features[b"schedule_seed"], features[b"time"], features[b"time_root"])

            fname = "pipeline_{:03d}_schedule_{:03d}_stages_{}.mp".format(pipe_seed, features[b"schedule_seed"], num_stages)
            with open(os.path.join(results_dir, fname), "wb") as fid:
              msgpack.dump(features, fid)
          except:
            if (completed - 1) % 1000 == 0:
              print(r, "failed")
          finally:
            completed += 1
  print("saved data to {}".format(results_dir))

  print(".Changing directory back to original location {}".format(curdir))
  os.chdir(curdir)


if __name__ == "__main__":
  # TODO: add mechanism to launch multiple
  parser = argparse.ArgumentParser()
  parser.add_argument("--results_dir", type=str, required=True)
  parser.add_argument("--workers", type=int, default=cpu_count(), help="number of workers for the parallel build")

  parser.add_argument("--evaluate", dest="evaluate", action="store_true", help="evaluate autoscheduler, instead of generating data samples")
  parser.add_argument("--predictor_url", type=str, default="tcp://localhost:5555", help="url of the throughput predictor server, useful when evaluating our predictions")
  parser.add_argument("--use_predictor_server", action="store_true", help="should we use the predictor server?")

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
  parser.add_argument("--hl_threads", type=int, default=8)

  # For distributed generation (to enseure that different nodes generate different
  # groups of programs)
  parser.add_argument("--node_id", type=int, default=0)
  parser.add_argument("--num_nodes", type=int, default=1)

  # Autoscheduler MachineParams
  parser.add_argument("--num_cores", type=int, default=cpu_count())
  # Size of last level cache (in bytes); default = 4096KB
  parser.add_argument("--llc_size", type=int, default=32768)
  parser.add_argument("--balance", type=int, default=40)

  parser.set_defaults(master_scheduler=False)

  parser.set_defaults(build_only=False)
  parser.set_defaults(bench_only=False)
  parser.set_defaults(gather_only=False)

  args = parser.parse_args()

  main(args)
