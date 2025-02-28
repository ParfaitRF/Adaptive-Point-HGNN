import os
import time
import argparse
import copy
from sys import getsizeof
from multiprocessing import Pool, Qeue, Process

import numpy as np
import torch

from data.kitti import KittiDataset
from models.graph_gen import get_graph_generate_fn
from models.models import get_model


def train(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if int(args.double_precision):  torch.set_default_dtype(torch.float64)
  if int(args.cuda) >= 0:  torch.cuda.manual_seed(args.seed)



