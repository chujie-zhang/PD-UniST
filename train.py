#!/usr/bin/python3

import argparse
import os
from trainer import CycPrompt_Trainer
import yaml

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream,Loader=yaml.FullLoader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/P2p.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)
    
    if config['name'] == 'PDPrompt':
        trainer = CycPrompt_Trainer(config)

        
    trainer.train()

###################################
if __name__ == '__main__':
    main()






