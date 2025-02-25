import numpy as np
import torch_directml
import torch
from torch.distributions import Exponential as dexp

# GEOMETRY

R = lambda yaw: np.array([[np.cos(yaw),  0,  np.sin(yaw)],
                      [0,            1,  0          ],
                      [-np.sin(yaw), 0,  np.cos(yaw)]])

# output image size
IMG_HEIGHT  = 375
IMG_WIDTH   = 1242

# DEEP LEARNING
GPU = torch_directml.device(0)

dexp = dexp(rate=1/torch.pi)
