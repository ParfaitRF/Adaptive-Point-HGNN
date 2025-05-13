import os
import time
import argparse
import copy
from sys import getsizeof
from multiprocessing import Pool, Queue, Process
from torch.utils.data import DataLoader

import numpy as np
import torch

#from models.models import get_model
from models.box_encoders_decoders import get_encoding_len
from util.config_util import load_config,save_config
from models.dataloader import CustomKittiDataset

os.chdir('..')

# ============================================================================ #
# ================================ SET PATHS ================================= #
# ============================================================================ #

script_dir = os.path.dirname(os.path.abspath(__file__))                         # get current script directory
train_config_path = os.path.join(                                               # get training config file path 
  script_dir, 'configs\\car_auto_T0_train_config'
)
config_path = os.path.join(script_dir, 'configs\\car_auto_T0_config')            # general config file path                       
dataset_root_dir    = os.path.join(script_dir, 'dataset\\kitti\\')                # dataset root directory  

# ============================================================================ #
# =========================== PARSE ARGUMENTS ================================ #
# ============================================================================ #

parser = argparse.ArgumentParser(description='Training of ACE-PGNN')            # add paths to parser
parser.add_argument('--train_config_path', type=str, default=train_config_path,
                   help='Path to train_config')
parser.add_argument('--config_path', type=str, default=config_path,
                   help='Path to config')
parser.add_argument('--dataset_root_dir', type=str, default=dataset_root_dir,
                   help='Path to KITTI dataset. Default="data\\kitti\\"')
parser.add_argument('--dataset_split_file', type=str,
                    default='',
                   help='Path to KITTI dataset split file.'
                   'Default="DATASET_ROOT_DIR/3DOP_splits'
                   '/train_config["train_dataset"]"')
args = parser.parse_args()

train_config = load_config(args.train_config_path)                              # load training configuration
DATASET_DIR = args.dataset_root_dir

if args.dataset_split_file == '':                                               # dataset split file path 
  DATASET_SPLIT_FILE = os.path.join(
    DATASET_DIR,'./3DOP_splits/'+train_config['train_dataset'])
else:
  DATASET_SPLIT_FILE = args.dataset_split_file

config = load_config(args.config_path)                                          # load general configuration


# ============================================================================ #
# =============================== DATA LOADER ================================ #
# ============================================================================ #

dataset = CustomKittiDataset(config=config,train_config=train_config)
NUM_CLASSES = dataset.num_classes                                               # number of classes

if ('NUM_TEST_SAMPLE' not in train_config) or (train_config['NUM_TEST_SAMPLE'] < 0):
  NUM_TEST_SAMPLE = dataset.num_files
else:
  NUM_TEST_SAMPLE = train_config['NUM_TEST_SAMPLE']

BOX_ENCODING_LEN  = get_encoding_len(config['box_encoding_method'])

dataloader = DataLoader(                                                        # initialize data loader
  dataset,
  batch_size=20,
  shuffle=True,
  num_workers=8,
  pin_memory=True
)








fetch_data(3464)