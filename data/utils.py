# =============================================================================#
# ================================== IMPORTS ==================================#
# =============================================================================#

from collections import defaultdict
from collections import namedtuple
import random
import numpy as np

from data.transformations import box3d_to_normals
from data.preprocess import (
  random_jitter,random_box_rotation,random_box_shift,random_transition,
  remove_background,random_rotation_all,random_flip_all,random_drop,
  random_global_drop,random_voxel_downsample,random_scale_all,
  random_box_global_rotation,dilute_background,empty
)


# =============================================================================#
# ============================== DATA AUGMENTATION ============================#
# =============================================================================#

AUG_METHOD_MAP = {
  'random_jitter':        random_jitter,
  'random_box_rotation':  random_box_rotation,
  'random_box_shift':     random_box_shift,
  'random_transition':    random_transition,
  'remove_background':    remove_background,
  'random_rotation_all':  random_rotation_all,
  'random_flip_all':      random_flip_all,
  'random_drop':          random_drop,
  'random_global_drop':   random_global_drop,
  'random_voxel_downsample':    random_voxel_downsample,
  'random_scale_all':     random_scale_all,
  'random_box_global_rotation': random_box_global_rotation,
  'dilute_background':    dilute_background,
}

def get_data_aug(aug_configs=[]):
  """ Get data augmentation function based on the configuration.
  @param aug_configs: a list of dictionaries containing the augmentation
                      configurations. 
  @return a function that applies the augmentation.
  """
  if len(aug_configs)==0: return empty

  def multiple_aug(cam_rgb_points, labels):
    for aug_config in aug_configs:
      aug_method = AUG_METHOD_MAP[aug_config['method_name']]
      cam_rgb_points, labels = aug_method(
        cam_rgb_points, labels, **aug_config['method_kwargs'])
    return cam_rgb_points, labels
  
  return multiple_aug


# =============================================================================#
# =========================== DOWNSAMPLING FUNCTIONS ==========================#
# =============================================================================#

def downsample_by_voxel(points:Points, voxel_size:float,method:str='AVERAGE',add_rnd3d:bool=False):
  """Downsample point cloud by voxel.

  @param points:      a Points namedtuple containing "xyz" and "attr".
  @param voxel_size:  the size of voxel cells used for downsampling.
  @param method:      'AVERAGE', all points inside a voxel cell are averaged
                      including xyz and attr.

  @return:  downsampled points and attributes.
  """

  if method == 'AVERAGE':   res = downsample_by_average_voxel(                  # downsample by averaging points in voxel
    points,voxel_size)  
  elif method == 'RANDOM':  res = downsample_by_random_voxel(                   # downsample by random voxel sizes
    points,voxel_size,add_rnd3d)
  else: raise Exception("Unknown method: %s" % method)                          # raise exception if method is unknown           
  
  return res


def downsample_by_average_voxel(points:Points, voxel_size:float):
  """Voxel downsampling using average function.
  
  @param points:      a Points namedtuple containing "xyz" and "attr".
  @param voxel_size:  the size of voxel cells used for downsampling.

  @return downsampled points and attributes.
  """

  xmin, ymin, zmin    = np.amin(points.xyz, axis=0)                             # get min values for each axis  
  xyz_offset  = np.asarray([[xmin, ymin, zmin]])                                # offset defined using min vals
  xyz_idx     = (points.xyz - xyz_offset) // voxel_size                         # define points voxel index
  xyz_idx     = xyz_idx.astype(np.int32)                                                      
  dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1                            # get grid dimensionss
  keys        = xyz_idx[:, 0] + (xyz_idx[:, 1] + xyz_idx[:, 2]*dim_y)*dim_x     # create unique voxel keys
  order       = np.argsort(keys)                                                                         
  keys        = keys[order]
  points_xyz  = points.xyz[order]
  unique_keys, lens   = np.unique(keys, return_counts=True)
  indices     = np.hstack([[0], lens[:-1]]).cumsum()
  downsampled_xyz     = np.add.reduceat(                                        # get downsampled points   
    points_xyz, indices, axis=0
  )/lens[:,np.newaxis]

  downsampled_attr    = None                                                    # downsample attributes if any exist
  if points.attr is not None:
    downsampled_attr  = np.add.reduceat(
      points.attr[order], indices, axis=0)/lens[:,np.newaxis]

  return Points(xyz=downsampled_xyz,attr=downsampled_attr)


def downsample_by_random_voxel(points:Points, 
                               voxel_size:float, add_rnd3d:bool=False):
  """Downsample the points using base_voxel_size at different scales and 
  randomly choosing a point in the voxel

  @param points:     a Points namedtuple containing "xyz" and "attr".
  @param voxel_size: base voxel size for downsampling.
  @param add_rnd3d:  add random noise to the voxel size.

  @return downsampled points and attributes.
  """

  xmin, ymin, zmin = np.amin(points.xyz, axis=0)                                # get min values for each axis               
  xyz_offset  = np.asarray([[xmin, ymin, zmin]])                                # offset defined using min vals

  xyz_idx     = (points.xyz - xyz_offset)                                       # define points voxel index
  if add_rnd3d:
    xyz_idx += voxel_size*np.random.random((1,3))
  xyz_idx     = xyz_idx // voxel_size

  dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1                            # get grid dimensionss
  keys        = xyz_idx[:, 0] + (xyz_idx[:, 1] + xyz_idx[:, 2]*dim_y)*dim_x     # create unique voxel keys
  voxels_idx  = defaultdict(list)                                               # dictionary containing voxel_key:point_indices  

  for pidx, key in enumerate(keys):                                             # assign points to voxels
    voxels_idx[key].append(pidx)
  voxels_idx  = dict(voxels_idx)           


  downsampled_xyz     = []                                                      # downsample points and attributes by random selection
  downsampled_attr    = []

  for key in voxels_idx:
    center_idx = random.choice(voxels_idx[key])
    downsampled_xyz.append(points.xyz[center_idx])
    downsampled_attr.append(points.attr[center_idx])

  return Points(xyz=np.array(downsampled_xyz),attr=np.array(downsampled_attr))


def sel_points_in_box3d(label:dict, points_xyz:np.array,
                        expend_factor:tuple=(1.0, 1.0, 1.0)):
  """ Filters points bounding box

  @param label:   a dictionary containing "x3d", "y3d", "z3d", "yaw",
                  "height", "width", "lenth".
  @param points:  a Points object containing "xyz" and "attr".
  @expend_factor: a tuple of (h, w, l) to expand the box.
  @return: a bool mask indicating points inside a 3D box.
  """

  normals, lower, upper = box3d_to_normals(label, expend_factor)                # get box normals
  projected   = np.matmul(points_xyz, np.transpose(normals))                    # project points to box normals
  points_in_x = np.logical_and(projected[:, 0] > lower[0],                      # create filters along all axis
                              projected[:, 0] < upper[0])
  points_in_y = np.logical_and(projected[:, 1] > lower[1],
                              projected[:, 1] < upper[1])
  points_in_z = np.logical_and(projected[:, 2] > lower[2],
                              projected[:, 2] < upper[2])
  mask        = np.logical_and.reduce((points_in_x, points_in_y, points_in_z))  # filter boxes

  return mask


def sel_points_in_box2d(label:str, points_xyz:np.array, expend_factor:tuple=(1.0, 1.0, 1.0)):
  """ Select points in a 2D (yz-plane) bounding box 

  @param label:        a dictionary containing "x3d", "y3d", "z3d", "yaw", 
                        "height", "width", "length".
  @param xyz:          a numpy array containing the points to be filtered.
  @param expend_factor:a tuple containing the scaling factors for the box.

  @return:             boolean mask indicating which points are within the bounding box
  """

  normals, lower, upper = box3d_to_normals(label, expend_factor)                # get normal vectors
  normals, lower, upper = normals[1:], lower[1:], upper[1:]                     
  projected   = np.matmul(points_xyz, np.transpose(normals))                    # project points onto normal planes
  points_in_y = np.logical_and(projected[:, 0] > lower[0],
                               projected[:, 0] < upper[0])
  points_in_z = np.logical_and(projected[:, 1] > lower[1],
                               projected[:, 1] < upper[1])
  mask = np.logical_and.reduce((points_in_y, points_in_z))                      # define point in y-z-plane bound

  return mask

