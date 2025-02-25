""" This file defines all functions needed for graph construction """

import time
import random

import numpy as np
from sklearn.neighbors import NearestNeighbors

import open3d

import torch
import torch_geometric as torchg
import torch_directml as dml

gpu = dml.device(0)


def multi_layer_downsampling(points_xyz,base_voxel_size,levels=[1],add_rnd3d=False):
  """ Downsamples the points using base_voxel_size at different scales 
  
  @param  points_xyz[array (N,3)]:  points
  @param  base_voxel_size[float]:   base voxel size
  @param  levels[list]:             list of levels
  @param  add_rnd3d[bool]:          add random 3d offset

  @return # TODO: what does this return?
  """
   
  xmin, ymin, zmin  = np.min(points_xyz, axis=0)        # get the minimum values of the points
  xyz_offset        = np.asarray([[xmin, ymin, zmin]])  # define min values as offset
  downsampled_list  = [points_xyz]                      # initialize downsampled list with original data
  last_level        = 0

  for level in levels: # TODO: what is a level (assuming scaling factors)
    if np.isclose(last_level,level): # no significant change in scale
      downsampled_list.append(np.copy(downsampled_list[-1]))
    else:
      if add_rnd3d: # rescaled grid with point adaptation
        xyz_idx = (points_xyz-xyz_offset+base_voxel_size*level*np.random((1,3))) // (base_voxel_size*level) # TODO: no idea what this does
        xyz_idx = xyz_idx.astype(np.int32)
        dim_x,dim_y,dim_z = np.max(xyz_idx,axis=0)+1
        keys          = xyz_idx[:,0] + xyz_idx[:,1]*dim_x + xyz_idx[:,2]*dim_x*dim_y
        sorted_order  = np.argsort(keys)
        sorted_keys   = keys[sorted_order]
        sorted_points_xyz = points_xyz[sorted_order]
        _,lens          = np.unique(sorted_keys,return_counts=True)
        indices         = np.hstack([[0],lens[:,-1]]).sumsum()
        downsampled_xyz = np.add.reduceat(
          sorted_points_xyz, indices, axis=0
        ) / lens[:,np.newaxis]
        downsampled_list.append(np.array(downsampled_xyz))
      else:
        pcd             = open3d.geometry.PointCloud()
        pcd.points      = open3d.utility.Vector3dVector(points_xyz)
        downsampled_xyz = pcd.voxel_down_sample(voxel_size=base_voxel_size*level).points
        downsampled_list.append(downsampled_xyz)
      last_level = level

  return downsampled_list


def multi_layer_downsampling_select(points_xyz, base_voxel_size, levels=[1],add_rnd3d=False):
  """ Downsample the points at different scales and match the downsampled points to original points by a nearest neighbor search
  
  @param  points_xyz[array (N,3)]:  points
  @param  base_voxel_size[float]:   base voxel size
  @param  levels[list]:             list of levels # TODO: meaning still unknown
  @param  add_rnd3d[bool]:          add random 3d offset

  @returns: downsampled_list, indices_list
  """

  # TODO: what does this return?p
  vertex_coord_list = multi_layer_downsampling(points_xyz, base_voxel_size, levels, add_rnd3d)
  num_levels = len(vertex_coord_list)
  assert num_levels == len(levels) + 1

  # match downsampled vertices to original by a nearest neighbor search.
  keypoint_indices_list = []
  last_level = 0

  for i in range(1,num_levels):
    current_level   = levels[i-1]
    base_points     = vertex_coord_list[i-1]
    current_points  = vertex_coord_list[i]

    if np.isclose(current_level,last_level): # TODO: why do we care about this?
      # same downsample scale (gnn layer)
      # just copy it, no need to search
      vertex_coord_list[i] = base_points
      keypoint_indices_list.append(
        np.expand_dims(np.arrange(base_points.shape[0]),axis=1))
    else:
      # different scale (pooling layer), search original points
      nbrs = NearestNeighbors( # TODO: understand the return variable of this function
        n_neighbors=1,algorithm='kd_tree',n_jobs=-1).fit(base_points) 

      indices = nbrs.kneighbors(current_points,return_distance=False)
      vertex_coord_list[i] = base_points[indices[:,0],:]
      keypoint_indices_list.append(indices)
    last_level = current_level
  return vertex_coord_list, keypoint_indices_list


def multi_layer_downsampling_random(points_xyz, base_voxel_size, levels=[1],
  add_rnd3d=False):
  """Downsample the points at different scales by randomly select a point within a voxel cell.

  @params points_xyz[array (N,3)]: points
  @params base_voxel_size[float]: base voxel size
  @params levels[list]: list of levels
  @params add_rnd3d[bool]: add random 3d offset

  returns: vertex_coord_list, keypoint_indices_list
  """
  xmax, ymax, zmax = np.amax(points_xyz, axis=0)
  xmin, ymin, zmin = np.amin(points_xyz, axis=0)
  xyz_offset = np.asarray([[xmin, ymin, zmin]])
  xyz_zeros = np.asarray([0, 0, 0], dtype=np.float32)
  vertex_coord_list = [points_xyz]
  keypoint_indices_list = []
  last_level = 0
  for level in levels:
    last_points_xyz = vertex_coord_list[-1]

    if np.isclose(last_level, level):
      # same downsample scale (gnn layer), just copy it
      vertex_coord_list.append(np.copy(last_points_xyz))
      keypoint_indices_list.append(
          np.expand_dims(np.arange(len(last_points_xyz)), axis=1))
    else:
      if not add_rnd3d:
        xyz_idx = (last_points_xyz - xyz_offset) // (base_voxel_size*level)
      else:
        xyz_idx = (last_points_xyz - xyz_offset +
            base_voxel_size*level*np.random.random((1,3))) // (base_voxel_size*level)
        
      xyz_idx     = xyz_idx.astype(np.int32)
      dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1
      keys        = xyz_idx[:, 0]+xyz_idx[:, 1]*dim_x+xyz_idx[:, 2]*dim_y*dim_x
      num_points  = xyz_idx.shape[0]

      voxels_idx  = {}
      for pidx in range(len(last_points_xyz)):
        key = keys[pidx]
        if key in voxels_idx: voxels_idx[key].append(pidx)
        else: voxels_idx[key] = [pidx]

      downsampled_xyz     = []
      downsampled_xyz_idx = []
      for key in voxels_idx:
        center_idx = random.choice(voxels_idx[key])
        downsampled_xyz.append(last_points_xyz[center_idx])
        downsampled_xyz_idx.append(center_idx)

      vertex_coord_list.append(np.array(downsampled_xyz))
      keypoint_indices_list.append(
        np.expand_dims(np.array(downsampled_xyz_idx),axis=1))
    last_level = level

  return vertex_coord_list, keypoint_indices_list

def gen_multi_level_local_graph_v3(
  points_xyz, base_voxel_size, level_configs, add_rnd3d=False,
  downsample_method='center'):
  """Generating graphs at multiple scale. This function enforce output
  vertices of a graph matches the input vertices of next graph so that
  gnn layers can be applied sequentially.

  @param points_xyz[array (NxD)]: N is the total number of the points. D isthe dimension of the coordinates.
  @param base_voxel_size[float]:  voxel size.
  @param level_configs[dict]:     dictionary of 'level', 'graph_gen_method','graph_gen_kwargs', 'graph_scale'.
  @param add_rnd3d[bool]:         whether to add random offset when downsampling.
  @param downsample_method[str]:  name of downsampling method.

  returns: vertex_coord_list, keypoint_indices_list, edges_list
  """
  if isinstance(base_voxel_size, list): # convert list to array
    base_voxel_size = np.array(base_voxel_size)
  # Gather the downsample scale for each graph
  scales = [config['graph_scale'] for config in level_configs] # get graph scales
  # Generate vertex coordinates
  if downsample_method=='center':
    vertex_coord_list, keypoint_indices_list = multi_layer_downsampling_select(
      points_xyz, base_voxel_size, scales, add_rnd3d=add_rnd3d)
  if downsample_method=='random':
    vertex_coord_list, keypoint_indices_list = multi_layer_downsampling_random(
      points_xyz, base_voxel_size, scales, add_rnd3d=add_rnd3d)
  # Create edges
  edges_list = []
  for config in level_configs:
    graph_level   = config['graph_level']
    gen_graph_fn  = get_graph_generate_fn(config['graph_gen_method'])
    method_kwarg  = config['graph_gen_kwargs']
    points_xyz    = vertex_coord_list[graph_level]
    center_xyz    = vertex_coord_list[graph_level+1]
    vertices      = gen_graph_fn(points_xyz, center_xyz, **method_kwarg)
    edges_list.append(vertices)

  return vertex_coord_list, keypoint_indices_list, edges_list


def gen_disjointed_rnn_local_graph_v3(
  points_xyz, center_xyz, radius, num_neighbors,
  neighbors_downsample_method='random',
  scale=None):
  """Generate a local graph by radius neighbors.
  """
  if scale is not None:
    scale = np.array(scale)           # convert scale list to array
    points_xyz = points_xyz/scale     # normalisation TODO: dont understand why we are not multiplying by scale here
    center_xyz = center_xyz/scale     # same
  nbrs = NearestNeighbors(            # init. nearest neighbor model
    radius=radius,algorithm='ball_tree', n_jobs=1, ).fit(points_xyz)
  indices = nbrs.radius_neighbors(center_xyz, return_distance=False) # TODO: makes little sense

  if num_neighbors > 0:
    if neighbors_downsample_method == 'random': # TODO: what if not?
      indices = [neighbors if neighbors.size <= num_neighbors else
        np.random.choice(neighbors, num_neighbors, replace=False)
        for neighbors in indices]

  vertices_v = np.concatenate(indices)
  vertices_i = np.concatenate(
    [i*np.ones(neighbors.size, dtype=np.int32) 
    for i, neighbors in enumerate(indices)])
  vertices = np.array([vertices_v, vertices_i]).transpose()
  return vertices


def get_graph_generate_fn(method_name):
  """ Get the graph generation function by method name """
  method_map = {
    'disjointed_rnn_local_graph_v3':gen_disjointed_rnn_local_graph_v3,
    'multi_level_local_graph_v3': gen_multi_level_local_graph_v3,
  }

  return method_map[method_name]




