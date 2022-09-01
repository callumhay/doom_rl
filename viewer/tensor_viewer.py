import torch
import torchshow as ts
import argparse


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Display a visualization of a given tensor.")
  parser.add_argument('-tensor', '--tensor', '-t', help="The saved tensor file to view.", default="")

  args = parser.parse_args()
  if len(args.tensor) == 0:
    parser.print_usage()
    exit()

  tensor = torch.load(args.tensor)
  ts.show(tensor)