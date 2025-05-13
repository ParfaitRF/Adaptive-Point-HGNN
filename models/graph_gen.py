"""
The file defines functions to generate graphs at multiple scales from a point 
cloud. 
"""

import random
from typing import Tuple, List
from collections import defaultdict

import numpy as np
from sklearn.neighbors import NearestNeighbors
from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector


def multi_layer_downsampling(points_xyz, base_voxel_size, levels=[0,1],method='random'):
  """
  Downsample the points using base_voxel_size at different scales.

  Parameters
  __________
  points_xyz: a [n, d] matrix. 
    n is the total number of the points. d is the dimension of the coordinates.
  base_voxel_size: float
    the cell size of voxel.
  levels: Tuple[float]
    list of level scales. The first element is the base voxel size.

  Returns
  ________
    vertex_idx_list: List[np.ndarray]
      list of downsampled point (indices) relative to original pcd.
  """

  if levels[0] != 0:
    levels = [0] + levels

  _range        = np.arange(points_xyz.shape[0])
  identity      = np.arange(points_xyz.shape[0])

  vertex_idx_list   = [identity]                                                # list of downsampled points (indices)

  def downsample_vertex(points):
    if method == 'random':
      return np.random.choice(points.shape[0])
    elif method == 'center':
      mean_point =  np.mean(points, axis=0)
      sq_dist = ((points-mean_point)**2).sum(axis=1)
      return np.argmin(sq_dist)
    else:
      raise ValueError(f"Unknown downsample method: {method}")
  
  for i,level in enumerate(levels):
    if i == len(levels) - 1: break

    if np.isclose(level,levels[i+1]):
      vertex_idx_list.append(np.copy(vertex_idx_list[-1]))
      #indices = vertex_idx_list[-1]
    else:
      pcd         = PointCloud()                                                # create point cloud object
      pcd.points  = Vector3dVector(points_xyz[vertex_idx_list[-1],:])           # set points          
      p3d_ds_points,_,point_to_voxel_map  = pcd.voxel_down_sample_and_trace(    # downsample points
        voxel_size=base_voxel_size*levels[i+1],
        min_bound=pcd.get_min_bound(),
        max_bound=pcd.get_max_bound()  
      )
      indices = np.array(                                                       # convert list to array             
        [list(vec) for vec in point_to_voxel_map], dtype=object
      )
      p3d_ds_points = np.asarray(p3d_ds_points.points)
      ds_points_idx = np.empty(p3d_ds_points.shape[0])                          

      for i in range(p3d_ds_points.shape[0]):                                   # compute downsample point (index)
        ds_points_idx[i] = indices[i][downsample_vertex(
          points_xyz[point_to_voxel_map[i],:]
        )]

      vertex_idx_list.append(ds_points_idx.astype(int))                         # append indices to list
  
  #vertex_idx_list[0] = vertex_idx_list[0][::, np.newaxis]                   
  return vertex_idx_list


def gen_disjointed_rnn_local_graph_v3(
  points_xyz:np.ndarray, center_xyz_idx:np.ndarray, 
  radius:float, num_neighbors:int=-1
) -> np.ndarray:
  """ Generates a disjointed local graph using base points and center points.

  Parameters
  ___________
  points_xyz: [n, d] matrix. 
    n is the total number of the points. d is the dimension of the coordinates.
  center_xyz: [m, d] matrix. 
    m is the total number of the centers. d is the dimension of the coordinates.
  radius: float
    radius for local graph generation.
  num_neighbors: int
    number of neighbors to sample. Default value is -1, which means all.
  scale: float or None
    scale factor for points and center coordinates.

  Returns
  ________
    vertices: [2, k] matrix
      k is the number of neighbors. The first row is the vertex index and the
      second row is the graph index.

  """

  center_xyz  = points_xyz[center_xyz_idx,:]                                    # get center points

  nbrs    = NearestNeighbors(                                                   # create FRNN search object
    radius=radius,algorithm='ball_tree', n_jobs=-1
  ).fit(points_xyz)
  indices = nbrs.radius_neighbors(center_xyz, return_distance=False)            # apply model to query points

  if (num_neighbors > 0):                                                       # truncate number of neighbors
    indices = [
      neighbors if neighbors.size <= num_neighbors else
      np.array(np.random.choice(neighbors, num_neighbors, replace=False))
      for neighbors in indices
    ]
      
  vertices_i  = np.concatenate(indices)                                         # compute vertex pairs   
  vertices_j  = np.concatenate([center_xyz_idx[j]*np.ones(neighbors.size, dtype=np.int32)
    for j, neighbors in enumerate(indices)]
  )
  vertices    = np.array([vertices_i, vertices_j]).transpose()

  return vertices


def get_graph_generate_fn(method_name):
  return {
    'disjointed_rnn_local_graph_v3':gen_disjointed_rnn_local_graph_v3,
    'multi_level_local_graph_v3': gen_multi_level_local_graph_v3,
  }[method_name]


def gen_multi_level_local_graph_v3(
  points_xyz, base_voxel_size, level_configs,downsample_method='center'
):
  """Generates graphs at multiple scale. This function enforces output
  vertices of a graph matches the input vertices of next graph so that
  gnn layers can be applied sequentially.

  Parameters
  ----------
    points_xyz: [n, d] matrix. 
      n is the total number of the points. d is the dimension of the coordinates.
    base_voxel_size: float 
      cell size of base voxel.
    level_configs: dict
      dictionary containing the settings for each layer
    downsample_method: string
      the name of downsampling method. Options are ['center','random'].
  
  Returns
  _______
    vertex_coord_list: List[np.ndarray]
      list of downsampled points indices relative to original pcd.
    edges_list: List[np.ndarray]
      list of edges for each layer.
  """
  
  scales = [0]+[config['graph_scale'] for config in level_configs]              # Gather the downsample scales
  
  vertex_coord_list = multi_layer_downsampling(
    points_xyz, base_voxel_size, scales,downsample_method
  )
    
  edges_list = []                                                               # Create edges
  for i,config in enumerate(level_configs):
    gen_graph_fn  = get_graph_generate_fn(config['graph_gen_method'])
    method_kwarg  = config['graph_gen_kwargs']
    vertices      = gen_graph_fn(
      points_xyz,
      vertex_coord_list[i+1],
      **method_kwarg
    )
    edges_list.append(vertices)

  return vertex_coord_list, edges_list