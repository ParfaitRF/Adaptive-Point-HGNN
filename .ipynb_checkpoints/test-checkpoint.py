import os
import random
from matplotlib import pyplot as plt

from data.kitti_dataset import KittiDataset

kitti = KittiDataset() # initialize data object

kitti.vis_crop_aug_sampler()