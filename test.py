import os

# curr_dir = os.getcwd()
# wd = os.path.join(curr_dir,'BHT - Apllied Mathematics\THESIS\code\Hyperbolic-Point-GNN\Workbench')
# os.chdir(wd)

from Workbench.dataset.kitti_dataset import KittiDataset


curr_dir = os.getcwd()
image_dir = os.path.join(curr_dir,'Workbench\\dataset\\kitti\\image')
point_dir = os.path.join(curr_dir,'Workbench\\dataset\\kitti\\velodyne')
calib_dir = os.path.join(curr_dir,'Workbench\\dataset\\kitti\\calib')
label_dir = os.path.join(curr_dir,'Workbench\\dataset\\kitti\\labels')

is_training = True

kd = KittiDataset(image_dir,point_dir,calib_dir,label_dir,is_training=is_training)

kd.get_velo_points(1)

# import torch
# import torch_directml

# GPU = torch_directml.device(0)
# CPU = torch.device('cpu')



