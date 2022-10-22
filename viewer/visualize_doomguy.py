import string
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils
import argparse

def vis_tensor(tensor, ch=0, all_kernels=False, nrow=8, padding=1):
  if len(tensor.shape) != 4: return
  n,c,w,h = tensor.shape

  if all_kernels: tensor = tensor.view(n*c, -1, w, h)
  elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

  rows = np.min((tensor.shape[0] // nrow + 1, 64))
  grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
  plt.figure(figsize=(nrow, rows))
  plt.imshow(grid.numpy().transpose((1,2,0)))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Display convolutional neural networks for a given checkpoint file.")
  parser.add_argument('-checkpoint', '--checkpoint', help="The checkpoint file to load.", default="")
  args = parser.parse_args()
  if len(args.checkpoint) == 0:
    parser.print_usage()
    exit()

  model = None
  state_dict = None
  weightKeys = None
  try:
    model = torch.jit.load(args.checkpoint)
    state_dict = model.state_dict()
    weightKeys = [ k for k in state_dict.keys() if k.find('online') != -1 and k.find('.weight') != -1 and k.find('running') == -1 and k.find('num_batches') ]
  except (RuntimeError):
    state_dict = torch.load(args.checkpoint)
    if "agent" in state_dict:
      state_dict = state_dict["agent"]
    elif "vae" in state_dict:
      state_dict = state_dict["vae"]
      #print(state_dict)
    weightKeys = [ k for k in state_dict.keys() if k.find('.weight') != -1 ]
  
  print(weightKeys)
  #biasKeys   = [ k for k in state_dict.keys() if k.find('.bias')   != -1 and k.find('conv2d') != -1 ]
  for w_key in weightKeys:
    #print(w_key)
    filter = state_dict[w_key].clone().cpu()
    #filter = filter - filter.min()
    #filter = filter / filter.max()
    vis_tensor(filter, all_kernels=False)
    plt.axis('off')
    plt.ioff()
    plt.title(label=w_key)

  plt.show()

    


