#!/usr/bin/env python3
"""Parametrized generation of random pipelines."""

import argparse
import time
import pdb
import generate_worker
from multiprocessing import cpu_count
from rq import Queue
from redis import Redis

def main(args):
  redis_conn = Redis("localhost", 6379, password=args.password)
  q = Queue(connection=redis_conn)

  for p in range(args.pipelines):
    q.enqueue(generate_worker.build, p, args.schedules, args.hl_threads)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--predictor_url", type=str, default="tcp://localhost:5555", help="url of the throughput predictor server, useful when evaluating our predictions")
  parser.add_argument("--use_predictor_server", action="store_true", help="should we use the predictor server?")

  parser.add_argument("--password", help="server password", type=str, required=True)

  # Generation params
  parser.add_argument("--pipelines", type=int, default=1000)
  parser.add_argument("--schedules", type=int, default=50)

  # For distributed generation (to enseure that different nodes generate different
  # groups of programs)
  parser.add_argument("--start_seed", type=int, default=0)

  # Autoscheduler MachineParams
  parser.add_argument("--hl_threads", type=int, default=8)

  args = parser.parse_args()

  main(args)
