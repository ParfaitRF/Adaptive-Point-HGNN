""" This file defines functionalities needed globally"""
import numpy as np
from collections import namedtuple

Points  = namedtuple('Points', ['xyz', 'attr'])                                 # data type for managing points/attributes

M_ROT   = lambda yaw: np.array([                                                # defines a rotation matrix based on the yaw angle
  [np.cos(yaw),  0,  np.sin(yaw)],                  
  [0,            1,  0          ],
  [-np.sin(yaw), 0,  np.cos(yaw)]])


# COLORS
COLOR1 = 1

OBJECT_COLORS = {
  'Pedestrian': ["DeepPink",(255,20,147)],
  'Person_sitting': ["DeepPink",(255,255,147)],
  'Car': ['Red', (255, 0, 0)],
  'Van': ['Red', (255, 255, 0)],
  'Cyclist': ["Salmon",(250,128,114)],
  'DontCare': ["Blue",(0,0,255)],
}
OCCLUSION_COLORS = [(0, 128, 0), (0, 255, 255), (0, 0, 128), (255, 255, 255)]

# IMAGE AND OBJET DIMENSIONS
IMG_WIDTH, IMG_HEIGHT = 1242, 376




# THRESHOLDS
OBJECT_HEIGHT_THRESHOLDS = [40, 25, 25]
TRUNCATION_THRESHOLDS = [0.15, 0.3, 0.5]
OCCULUSION_THRESHOLDS = [0, 1, 2]                                               # 0=visible 1=partly occluded, 2=fully occluded, 3=unknown
