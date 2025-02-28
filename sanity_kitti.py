import os
import random

from data.kitti_dataset import KittiDataset

kitti = KittiDataset()

# dataset initialization and statistics retrieval
# print(kitti) 

# visualisation/ transformations check
idx = random.choice(range(kitti.num_files))
print(idx)
kitti.inspect_points(frame_idx=idx,downsample_voxel_size=0.05)