import os     # for file handline
import time   # nomen est domen
from os.path import isfile, join                # file path interactions
import numpy  as np                             # business as usual
import random # nomen est domen
import open3d # for 3D data handling
import cv2    # for image processing and ml
from collections import namedtuple,defaultdict  # nomen es domen
from globals import R                           # rotation matrix

Points = namedtuple('Points',['xyz','attr'])    # stores set of 3D points

def downsample_by_average(points,voxel_size):
  """ Downsamples point set using voxels of given size using average point corrd. in voxel

  @param points[Points]:    3D point tuple to be downsampled
  @param voxel_size[float]: desired voxel size

  @return [Points]:           downsampled Points object 
  """

  # create voxel grid
  xmax,ymax,zmax  = np.amax(points, axis=0)         # max cloud values along all axes
  xmin,ymin,zmin  = np.amin(points, axis=0)         # min cloud values along all axes
  o               = np.zeros(3,dtype=np.float32)    # origin
  xyz_offset      = np.array([xmin,ymin,zmin])      # cloud offset from origin
  xyz_idx         = np.asarray(points.xyz - xyz_offset,dtype=np.float32) // voxel_size # point voxel indezes
  dim_x,dim_y,dim_z = np.amax(xyz_idx,axis=0) + 1   # gird dimensions
  voxel_keys      = np.asarray(xyz_idx[:,0] + xyz_idx[:,1]*dim_x + xyz_idx[:,2]*dim_x*dim_y,dtype=np.int32) # voxel keys
  voxel_order     = np.argsort(voxel_keys)           # sort points by voxel key
  vocel_keys      = voxel_keys[voxel_order]                     # sorted voxel keys
  points_xyz      = points.xyz[voxel_order]                     # sorted point coordinates
  unique_keys,lens = np.unique(voxel_keys,return_counts=True)   # unique voxel keys and their counts
  indices         = np.hstack([[0],lens[:-1]]).cumsum()         # commulative voxel population
  downsampled_xyz = np.add.reduceat(                # get voxel average points
    points_xyz,indices,axis=0
  ) / lens[:,np.newaxis]                    
           
  downsampled_attr  = None
  if points.attr is not None:                                  # if attributes are present                 
    attr = points.attr[voxel_order]
    downsampled_attr = np.add.reduceat(
      attr, indices,axis=0
    )  / lens[:,np.newaxis]
  
  return Points(xyz=downsampled_xyz,attr=downsampled_attr)


def downsample_by_random(points,voxel_size,add_rnd3d=False):
  """ Downsamples point set using voxels of given size usgin random point in voxel as representative

  @param points[Points]:    3D point tuple to be downsampled
  @param voxel_size[float]: desired voxel size 
  @param add_rnd3d[bool]:   add random 3D points to fill empty voxels

  @return [Points]:           downsampled Points object
  """

  # create voxel grid
  xmax,ymax,zmax  = np.amax(points, axis=0)         # max cloud values along all axes
  xmin,ymin,zmin  = np.amin(points, axis=0)         # min cloud values along all axes
  o               = np.zeros(3,dtype=np.float32)    # origin
  xyz_offset      = np.array([xmin,ymin,zmin])      # cloud offset from origin

  # conditionally scales cloud to fit desired voxel size
  xyz_idx         = np.asarray(
      points.xyz - xyz_offset,
      dtype=np.float32) // voxel_size

  if add_rnd3d:                                     # scale cloud to fit voxel size    
    xyz_idx  += voxel_size*np.random((1,3)) // voxel_size
  
  dim_x, dim_y, dim_z = np.amax(xyz_idx,axis=0) + 1 # gird dimensions
  voxel_keys          = np.asarray(                 # voxel keys 
    xyz_idx[:,0] + xyz_idx[:,1]*dim_x + xyz_idx[:,2]*dim_x*dim_y,
    dtype=np.int32)
  num_points          = xyz_idx.shape[0]

  # dictionary to contain keys and corresponding point indices
  voxel_idxs  = {int(key): [] for key in np.unique(voxel_keys)} 
  for point_idx in range(len(points.xyz)):
    voxel_idxs[voxel_keys[point_idx]].add(point_idx)

  downsampled_xyz   = []
  downsampled_attr  = []

  for key in voxel_idxs.keys():
    # select random point from voxel as representative
    center_idx = random.choice(voxel_idxs[key])  
    downsampled_xyz.append(points.xyz[center_idx])
    downsampled_attr.append(points.attr[center_idx])

  return Points(xyz=np.asarray(downsampled_xyz),attr=np.asarray(downsampled_attr))   


def downsample_by_voxel(points,voxel_size,method="AVERAGE"):
  """ Downsample point set using voxel grid

  @param points[Points]:    3D point tuple to be downsampled
  @param voxel_size[float]: desired voxel size
  @param method[str]:       downsampling method ["AVERAGE","RANDOM"]

  @return [Points]:           downsampled Points object
  """

  # create voxel grid
  xmax,ymax,zmax  = np.amax(points, axis=0)         # max cloud values along all axes
  xmin,ymin,zmin  = np.amin(points, axis=0)         # min cloud values along all axes
  dim_x = int((xmax - xmin) / voxel_size + 1)
  dim_y = int((ymax - ymin) / voxel_size + 1)
  dim_z = int((zmax - zmin) / voxel_size + 1)
  voxel_account = {}
  xyz_idx = np.int32(                               # 3D point -> voxel indices
    (points.xyz - np.array([xmin,ymin,zmin])) / voxel_size
  )

  for pidx, (x_idx, y_idx, z_idx) in enumerate(xyz_idx):
    # TODO: checkk bug impact
    key = x_idx + y_idx*dim_x + z_idx*dim_x*dim_y
    if key in voxel_account:
      voxel_account[key] = [pidx]
    else:
      voxel_account[key].append(pidx)

    # compute downsampled representation
    downsampled_xyz_list = []
    if points.attr is not None:
      downsampled_attr_list = []
    if method == "AVERAGE":
      for idx,pidx_list in voxel_account.iteritems():
        if len(pidx_list):
          downsampled_xyz_list.append(np.mean(points.xyz[pidx_list,:],axis=0, keepdims=True))
        if points.attr is not None:
          downsampled_attr_list.append(np.mean(points.attr[pidx_list],axis=0))
          if points.attr is not None:
            downsampled_attr_list = np.mean(points.attr[pidx_list,:],axis=0,keepdims=True)


    if points.attr is not None:
      attributes = np.vstack(downsampled_attr_list)

    return Points(xyz=np.vstack(downsampled_xyz_list),attr=attributes)
  

def box3d_to_viewpoint(label,expend_factor=(1.0,1.0,1.0)):
  """ Convert bounding box corrdinates to viewpoint(camera) corrdinates

  @param label[dict]: dictionary containing "x3d", "y3d", "z3d", "yaw", "height", "width", "length".
  @param expend_factor[tuple]: 3D scale factor

  @return Points:           viewpoint cube corner representation of bounding box of size (8,3)
  """

  M_rot   = R(label['yaw'])
  
  h   = label['height'] 
  dh  = h * (expend_factor[0] - 1)
  w   = label['width']*expend_factor[1]
  l   = label['length']*expend_factor[2]
  corners = np.array([[l,dh,w],           # front up right
    	                [l,dh,-w],          # front up left
                      [-l,dh,-w],         # back up left
    	                [-l,dh,w],          # back up right   
                      [l,-2*h-dh,w],      # front down right
    	                [l,-2*h-dh,-w],     # front down left
                      [-l,-2*h-dh,-w],    # back down left
    	                [-l,-2*h-dh,w]])/2  # back down right
  
  r_corners = corners.dot(np.transpose(M_rot))  # rotate around yaw axis
  # displacement from camera
  tx = label['x3d']
  ty = label['y3d']
  tz = label['z3d']
  cam_coords = r_corners + np.array([tx,ty,tz])  # translate relative to camera

  return Points(xyz = cam_coords,attr = None)


def box3d_to_normals(label,expend_factor=(1.0,1.0,1.0)):
  """ Projet 3D box into camera coordinates, compute box center and normals

  @param label[dict]: dictionary containing "x3d", "y3d", "z3d", "yaw", "height", "width", "length".
  @param expend_factor[tuple]: 3D scale factor

  @return [tuple]: tuple of [3dbox,center,normals]
  """

  box3d_points = box3d_to_viewpoint(label,expend_factor)  # get viewpoint representation of box
  box3d_points_xyz = box3d_points.xyz                      # box corner coordinates

  wx = box3d_points_xyz[[0],:] - box3d_points_xyz[[4],:]           # front width vector
  lx = np.matmul(wx,box3d_points_xyz[4,:])
  ux = np.matmul(wx,box3d_points_xyz[0,:])

  wy = box3d_points_xyz[[0],:] - box3d_points_xyz[[1],:]           # front depth vector
  ly = np.matmul(wx,box3d_points_xyz[1,:])
  uy = np.matmul(wx,box3d_points_xyz[0,:])

  wz = box3d_points_xyz[[0],:] - box3d_points_xyz[[3],:]           # front height vector
  lz = np.matmul(wx,box3d_points_xyz[3,:])
  uz = np.matmul(wx,box3d_points_xyz[0,:])

  return (np.concatenate([wx,wy,wz],axis=0),
          np.concatenate([lx,ly,lz]),
          np.concatenate([ux,uy,uz]))



  







