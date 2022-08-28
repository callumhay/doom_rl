import torch
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Display the difference between two neural network checkpoint files.")
  parser.add_argument('-n1', '--n1')
  parser.add_argument('-n2', '--n2')
  args = parser.parse_args()
  if len(args.n1) == 0 or len(args.n2) == 0:
    parser.print_usage()
    exit()
  
  model1 = torch.jit.load(args.n1)
  model2 = torch.jit.load(args.n2)

  state_dict1 = model1.state_dict()
  state_dict2 = model2.state_dict()

  # Check the weights...
  weightKeys = [ k for k in zip(state_dict1.keys(), state_dict2.keys()) if k[0].find('.weight') != -1 and k[1].find('.weight') != -1 ]
  for k in weightKeys:
    w1 = state_dict1[k[0]]
    w2 = state_dict2[k[1]]
    diff = w1 - w2
    print(diff)
