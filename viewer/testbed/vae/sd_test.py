import yaml
from yaml.loader import SafeLoader

from sd_vae import SDVAE
from torchsummary import summary

from doom_env import PREPROCESS_FINAL_SHAPE_C_H_W
from doom_agent_thread import DoomAgentThread

if __name__ =="__main__":
  with open("config/sd_vae_config.yaml", "r") as ymlfile:
    vae_config = yaml.load(ymlfile, Loader=SafeLoader)
    
  sd_vae = SDVAE(vae_config['sd_vae_config'], vae_config['embed_dim']).cuda()
  summary(sd_vae, input_size=PREPROCESS_FINAL_SHAPE_C_H_W, device="cuda")
  
  doom_agent = DoomAgentThread()
  doom_agent.run()