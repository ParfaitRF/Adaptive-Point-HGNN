""" This file defines functionalities needed globally"""
import numpy as np
from collections import namedtuple
import torch


Points  = namedtuple('Points', ['xyz', 'attr'])                                 # data type for managing points

LABEL_MAP = {
  obj : 2*i+1 for i, obj in enumerate([
  'KA','Background','Pedestrian','Person_sitting','Cyclist',
  'Car', 'Van', 'Truck', 'Misc','DontCare'
  ])
}

M_ROT   = lambda yaw: np.array([                                                # defines a rotation matrix based on the yaw angle
  [np.cos(yaw),  0,  np.sin(yaw)],                  
  [0,            1,  0          ],
  [-np.sin(yaw), 0,  np.cos(yaw)]])

BOX_OFFSET = lambda l,w,h,delta_h: np.array([                                   # defines untilted bbox 
  [ l/2,  -h/2-delta_h/2,  w/2],
  [ l/2,  -h/2-delta_h/2, -w/2],
  [-l/2,  -h/2-delta_h/2, -w/2],
  [-l/2,  -h/2-delta_h/2,  w/2],

  [ l/2, delta_h/2, 0],
  [ -l/2, delta_h/2, 0],
  [l/2, -h-delta_h/2, 0],
  [-l/2, -h-delta_h/2, 0],

  [0, delta_h/2, w/2],
  [0, delta_h/2, -w/2],
  [0, -h-delta_h/2, w/2],
  [0, -h-delta_h/2, -w/2]]
)

# COLORS
COLOR1 = 1
COLORS = [                                                                      # color list
  ["Olive",(0,128,0)],
  ["Grey",(155,155,155)],
  ["DeepPurple",(205, 0, 255)],
  ["DeepPink",(255,255,147)],
  ["Salmon",(250,128,114)],
  ['Red', (255, 0, 0)],
  ['Yellow', (255, 255, 0)],
  ['Orange', (255, 150, 0)],
  ['Cyan', (0, 255, 255)],
  ["Blue",(0,0,255)],
  ["ForestGreen", (34, 139, 34)],
  
]
COLOR_MAP = {                                                                   # object color map
  k:v for k,v in zip(
  LABEL_MAP.keys(), COLORS)
}
OCCLUSION_COLORS = [(0, 128, 0), (0, 255, 255), (0, 0, 128), (255, 255, 255)]

# IMAGE AND OBJECT DIMENSIONS
IMG_WIDTH, IMG_HEIGHT = 1242, 376

# THRESHOLDS
OBJECT_HEIGHT_THRESHOLDS  = [40, 25, 25]
TRUNCATION_THRESHOLDS     = [0.15, 0.3, 0.5]
OCCULUSION_THRESHOLDS     = [0,1,2]                                             # 0=visible 1=partly occluded, 2=fully occluded, 3=unknown

# MODEL RELATED GLOBALS
device = 'cuda' if torch.cuda.is_available() else 'cpu'                         # set device to cuda if available
