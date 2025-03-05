""" Contains all fucntions needed for coordinate transformations """
import numpy as np
from globals import M_ROT,Points


def box3d_to_cam_points(label:tuple, expend_factor:list=(1.0, 1.0, 1.0)):
  """ Project 3D box form velodyne coordinates into camera coordiantes 
  
  @param label:         a dictionary containing "x3d", "y3d", "z3d", "yaw", 
                        "height", "width", "length".
  @param expend_factor: a tuple containing the scaling factors for the box.

  @return:              a Points namedtuple containing the camera coordinates.        
  """

  yaw = label['yaw']

  R = M_ROT(yaw)                                                                # get rotation matrix
  h = label['height']                                                           # get label dimensions
  delta_h = h*(expend_factor[0]-1)
  w = label['width']*expend_factor[1]
  l = label['length']*expend_factor[2]
  corners = np.array([
    [ l/2,  delta_h/2,  w/2],     # front up right
    [ l/2,  delta_h/2, -w/2],     # front up left
    [-l/2,  delta_h/2, -w/2],     # back up left
    [-l/2,  delta_h/2,  w/2],     # back up right
    [ l/2, -h-delta_h/2,  w/2],   # front down right
    [ l/2, -h-delta_h/2, -w/2],   # front down left
    [-l/2, -h-delta_h/2, -w/2],   # back down left
    [-l/2, -h-delta_h/2,  w/2]])  # back down right
  
  r_corners = corners.dot(np.transpose(R))                                      # rotated corners
  tx,ty,tz = label['x3d'],label['y3d'],label['z3d']                             # get translation values
  cam_points_xyz = r_corners+np.array([tx, ty, tz])                             # translate corner by center coordinates
  
  return Points(xyz = cam_points_xyz, attr = None)


def box3d_to_normals(label:dict, expend_factor:tuple=(1.0, 1.0, 1.0)):
  """ Project a 3D box into camera coordinates, compute the center
  of the box and normals.

  @param label:         a dictionary containing "x3d", "y3d", "z3d", "yaw", 
                        "height", "width", "length".
  @param expend_factor: a tuple containing the scaling factors for the box.

  @return:              a Points namedtuple box local normals.  
  """

  box3d_points      = box3d_to_cam_points(label, expend_factor)                 # get camera system Points object       
  box3d_points_xyz  = box3d_points.xyz                                          # get camera system coordinates

  wx = box3d_points_xyz[[0], :] - box3d_points_xyz[[4], :]                      # compute normal vectors
  lx = np.matmul(wx, box3d_points_xyz[4, :])
  ux = np.matmul(wx, box3d_points_xyz[0, :])
  wy = box3d_points_xyz[[0], :] - box3d_points_xyz[[1], :]                      # get lower bounds
  ly = np.matmul(wy, box3d_points_xyz[1, :])
  uy = np.matmul(wy, box3d_points_xyz[0, :])
  wz = box3d_points_xyz[[0], :] - box3d_points_xyz[[3], :]                      # get upper bounds                  
  lz = np.matmul(wz, box3d_points_xyz[3, :])
  uz = np.matmul(wz, box3d_points_xyz[0, :])

  return(np.concatenate([wx, wy, wz], axis=0),
    np.concatenate([lx, ly, lz]), np.concatenate([ux, uy, uz]))


def boxes_3d_to_corners(boxes_3d:list):
  """ Translates bounding boxes from origin to place in R^3 and applies yaw tilt.

  @param boxes_3d:  list of bounding boxes in 3D

  @returns:          list of translated bounding boxes
  """

  all_corners = []

  for x3d, y3d, z3d, l, h, w, yaw in boxes_3d:
    corners = np.array([                                                        # define bounding box corners                    
      [ l/2,  0.0,  w/2], # front up right
      [ l/2,  0.0, -w/2], # front up left
      [-l/2,  0.0, -w/2], # back up left
      [-l/2,  0.0,  w/2], # back up right
      [ l/2, -h,  w/2],   # front down right
      [ l/2, -h, -w/2],   # front down left
      [-l/2, -h, -w/2],   # back down left
      [-l/2, -h,  w/2]])  # back down right
    R = M_ROT(yaw)                                                              # get rotation matrix
    r_corners = corners.dot(np.transpose(R))                                    # rotate corners
    cam_points_xyz = r_corners+np.array([x3d, y3d, z3d])                        # translates cube from origin    
    all_corners.append(cam_points_xyz)                                          # append to list     

  return np.array(all_corners) 



def cam_points_to_image(points, calib):
  """Convert velodyne points to image plane points.

  @param calib: a dictionary containing calibration information.
  @returm:  points on image plane: a [M, 2] float32 numpy array,
            a mask indicating points: a [N, 1] boolean numpy array.
  """

  cam_points_xyz1 = np.hstack([points.xyz, np.ones([points.xyz.shape[0],1])])   # homogeneous coordinates
  img_points_xyz  = np.matmul(                                                  # transform to camera coordinates
    cam_points_xyz1, np.transpose(calib['cam_to_image']))
  img_points_xy1  = img_points_xyz/img_points_xyz[:,[2]]                        # project onto camera plane ??
  img_points      = Points(img_points_xy1, points.attr)                         # convert into Points object

  return img_points



def velo_to_cam(points, calib):
  """ Convert points in velodyne coordinates to camera coordinates using 
      homogeneous coordinates.

  @param points_xyz: a Points object xyz attribute
  @param calib: a dictionary containing calibration information.
  @return: a Points object containing points in camera coordinates.
  """

  velo_xyz1 = np.hstack([points.xyz, np.ones([points.xyz.shape[0],1])])         # convert to homogeneous coordinates  
  cam_xyz = np.transpose(
    np.matmul(calib['velo_to_cam'], np.transpose(velo_xyz1))[:3, :])            # transform to camera coordinates
  
  return Points(xyz=cam_xyz,attr=points.attr)


def cam_to_velo(points_xyz, calib):
  """ Convert points in camera coordinates to velodyne coordinates using 
      homogeneous coordinates.

  @param points_xyz:  a Points object xyz attribute
  @param calib:       a dictionary containing calibration information.
  @return: a Points object containing points in velodyne coordinates.
  """
  cam_xyz1 = np.hstack([points_xyz, np.ones([points_xyz.shape[0],1])])          # convert to homogeneous coordinates
  velo_xyz = np.matmul(cam_xyz1, np.transpose(calib['cam_to_velo']))[:,:3]      # transform to velodyne coordinates

  return velo_xyz