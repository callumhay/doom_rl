from email.policy import default
import os
import sys
import argparse
import time
import random
import inspect
from distutils.util import strtobool

import vizdoom as vzd

import numpy as np
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from vae import SDVAE

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, "doom_ppo"))
import doom_ppo


def parse_args():
  parser = argparse.ArgumentParser()
  
  # Top-level training arguements
  parser.add_argument("--model", type=str, default=None, help="Preexisting model to load (.chkpt file)")
  parser.add_argument("--save-steps", type=int, default=500000, help="Steps between network saves")
  parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate of the optimizer")
  parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
  parser.add_argument("--seed", type=int, default=1, help="RNG seed")
  parser.add_argument("--torch-deterministic", type=lambda x:bool(strtobool(x)), 
    default=True, nargs="?", const=True, help="Setting for `torch.backends.cudnn.deterministic`")
  
  # Network parameters
  parser.add_argument("--starting-channels", type=int, default=32, help="Initial number of channels in the convolutional networks")
  parser.add_argument("--num-res-blocks", type=int, default=1, help="Number of ResNet blocks in the convolutional networks")
  parser.add_argument("--ch-mult", type=str, default="1,2,2", 
    help="Multipliers of '--starting-channels', for the number of channels for each layer of the convolutional networks")
  parser.add_argument("--dropout", type=float, default=0.0, help="Dropout percentage for the ResNet blocks in the convolutional networks")
  parser.add_argument("--z-channels", type=int, default=10, help="Number of z-channels for the output of the encoder before embedding")
  parser.add_argument("--embed-dim", type=int, default=4, help="Number of embedding dimensions (non-categorical only)")
  parser.add_argument("--kl-weight", type=float, default=0.1, help="Weighting/Coefficient of VAE KL Loss")
  parser.add_argument("--max-grad-norm", type=float, default=0.5, help="The maximum norm for the gradient clipping")

  # Categorical VAE parameters
  parser.add_argument("--categorical", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True, help="Whether to use a categorical latent")
  parser.add_argument("--num-categories", type=int, default=96, help="Number of categories in the VAE latent distribution")
  parser.add_argument("--num-classes", type=int, default=64, help="Number of classes in the VAE latent distribution")

  # Game specific args
  parser.add_argument("--map", type=str, default="E1M2", help="Map(s) to load for play/training")

  args = parser.parse_args()
  args.use_labels_buffer = True
  args.ch_mult = [int(item) for item in args.ch_mult.split(',')]

  return args


if __name__ == "__main__":
  args = parse_args()

  # Seeding
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = args.torch_deterministic
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  env = doom_ppo.DoomEnv(args, 1000000, True, vzd.ScreenResolution.RES_160X120, 60, 80)
  vae = SDVAE(env, args).to(device)
  #vae = CategoricalVAE(Encoder(args, env.observation_shape()), Decoder(args, env.observation_shape())).to(device)
  from torchsummary import summary
  summary(vae, input_size=env.observation_shape(), device=device.type)
  optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate, betas=(0.5,0.9))
  
  init_step = 0
  global_step = 0
  num_batches = 0
  
  if args.model is not None:
    print(f"Model file '{args.model}' found, loading...")
    model_dict = torch.load(args.model)
    init_step = model_dict["timesteps"]
    global_step = init_step
    num_batches = model_dict["numbatches"]
    try:
      vae.load_state_dict(model_dict["vae"], strict=False)
    except RuntimeError as e:
      print("Could not load vae network:")
      print(e)
    try:
      optimizer.load_state_dict(model_dict["optim"])
    except RuntimeError as e:
      print("Could not load optimizer:")
      print(e)
    print("Model loaded!")
  else:
    print(f"Could not find/load model file '{args.model}'")
  
  run_name =  f"doom_vae_{args.seed}_{int(time.time())}"
  run_dir = os.path.join("runs", run_name)
  writer = SummaryWriter(run_dir)
  writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
  )

  curr_batch = []
  num_updates = 100000000
  
  env.reset()
  for update in range(num_updates):
    global_step += 1
    
    frac = 1.0 - (update - 1.0) / num_updates
    new_lr = frac * args.learning_rate
    optimizer.param_groups[0]['lr'] = new_lr
    
    next_obs, _, _, _, _ = env.step(env.random_action())
    curr_batch.append(torch.tensor(next_obs, dtype=torch.float32).to(device))
    if len(curr_batch) >= args.batch_size:
      x = torch.stack(curr_batch).to(device)
      loss, reconst_loss, kld_loss, x_hat, posterior = vae.training_step(x)
      
      optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
      optimizer.step()
      
      curr_batch = []
      num_batches += 1
      
      writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
      writer.add_scalar("losses/reconstruction_loss", reconst_loss.item(), global_step)
      #writer.add_scalar("losses/reconstruction_rgb_loss", reconst_rgb_loss.item(), global_step)
      #writer.add_scalar("losses/reconstruction_label_loss", reconst_label_loss.item(), global_step)
      writer.add_scalar("losses/kld_loss", kld_loss.item(), global_step)
      writer.add_scalar("losses/total_loss", loss.item(), global_step)

      if num_batches % 25 == 0:
        writer.add_images("imgs/rgb_obs_reconst", torch.stack([x[0,0:3], x_hat[0,0:3]]), global_step)
        writer.add_images("imgs/bad_stuff_obs_reconst", torch.stack([x[0,3:4],x_hat[0,3:4]]), global_step)
        writer.add_images("imgs/good_stuff_obs_reconst", torch.stack([x[0,4:5],x_hat[0,4:5]]), global_step)
        writer.add_images("imgs/utility_stuff_obs_reconst", torch.stack([x[0,5:6],x_hat[0,5:6]]), global_step)

      #if num_batches % 1000 == 0:
      #  writer.add_embedding(tag="model/posterior_mean", mat=posterior.mean.view(args.batch_size,-1), global_step=global_step)
      #  writer.add_embedding(tag="model/posterior_logvar", mat=posterior.logvar.view(args.batch_size,-1), global_step=global_step)
        
    if global_step % args.save_steps == 0:
      save_dict = {
        "timesteps": global_step,
        "numbatches": num_batches,
        "vae": vae.state_dict(),
        "optim": optimizer.state_dict()
      }
      save_path = os.path.join(run_dir, f"doom_vae_{global_step}.chkpt")
      torch.save(save_dict, save_path)