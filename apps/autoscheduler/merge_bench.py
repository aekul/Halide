import pdb
import argparse
import os
import msgpack

def merge_bench(data_dir, id, num_stages, time):
  with open("{}/{}.mp".format(data_dir, id), "rb") as f:
    data = msgpack.load(f)

  data[b"id"] = id
  data[b"num_stages"] = num_stages
  data[b"time"] = time

  with open("{}/{}.mp".format(data_dir, id), "wb") as f:
    msgpack.dump(data, f)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_dir", help="path to data directory", type=str, required=True)
  parser.add_argument("--id", help="ID of sample", type=str, required=True)
  parser.add_argument("--num_stages", help="number of stages in pipeline", type=int, required=True)
  parser.add_argument("--time", help="time", type=float, required=True)
  args = parser.parse_args()
  data_dir = args.data_dir
  id = args.id
  num_stages = args.num_stages
  time = args.time

  merge_bench(data_dir, id, num_stages, time)
