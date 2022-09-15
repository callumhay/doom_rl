#!/usr/bin/env python3

import argparse
from doom_agent_thread import DoomAgentThread, DoomAgentOptions

if __name__ =="__main__":
  opts = DoomAgentOptions()
  
  parser = argparse.ArgumentParser(description="Train a Doom VAE network.")
  parser.add_argument('-net', '--net', '--n', '-n', help="An existing VAE model network to load.", default=opts.model_filepath)
  parser.add_argument('-map', '--map', '--m', '-m', help="The Doom map to load.", default=opts.doom_map)
  args = parser.parse_args()

  opts.doom_map = args.map
  opts.model_filepath = args.net
  
  doom_agent = DoomAgentThread(opts)
  doom_agent.run()