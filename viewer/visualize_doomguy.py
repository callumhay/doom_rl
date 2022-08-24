import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils

def vis_tensor(tensor, ch=0, all_kernels=False, nrow=8, padding=1):
  n,c,w,h = tensor.shape

  if all_kernels: tensor = tensor.view(n*c, -1, w, h)
  elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

  rows = np.min((tensor.shape[0] // nrow + 1, 64))
  grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
  plt.figure(figsize=(nrow, rows))
  plt.imshow(grid.numpy().transpose((1,2,0)))


if __name__ == "__main__":
  model = torch.jit.load("../build/checkpoints/2022-08-23T18-46-08/doomguy_net_v1_save_1.chkpt")
  state_dict = model.state_dict()
  weightKeys = [ k for k in state_dict.keys() if k.find('.weight') != -1 and k.find('conv2d') != -1]

  for w_key in weightKeys:
    #print(w_key)
    filter = state_dict[w_key].clone()
    filter = filter - filter.min()
    filter = filter / filter.max()
    vis_tensor(filter, all_kernels=False)
    plt.axis('off')
    plt.ioff()
    plt.title(label=w_key)

  plt.show()

    


