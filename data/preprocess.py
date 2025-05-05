"""This file defines functions to augment data from dataset. """
import warnings
import numpy as np
import random
from copy import deepcopy

from globals import M_ROT,Points
from data.utils import (sel_points_in_box3d,downsample_by_voxel)
from data.transformations import boxes_3d_to_corners
from utils.nms import overlapped_boxes_3d_fast_poly


def random_jitter(
  cam_rgb_points:Points, labels:dict, xyz_std:tuple=(0.1, 0.1, 0.1)):
  """ Adds random noise to points coordinates

  @param cam_rgb_points: a Points object containing "xyz" and "attr".
  @param labels:         a dictionary containing the bounding box labels.
  @param xyz_std:        a tuple containing the standard deviation for the noise.

  return: a tuple of (Points, dict) containing the jittered points and labels. 
  """
  xyz     = cam_rgb_points.xyz
  x_delta = np.random.normal(size=(xyz.shape[0], 1), scale=xyz_std[0])
  y_delta = np.random.normal(size=(xyz.shape[0], 1), scale=xyz_std[1])
  z_delta = np.random.normal(size=(xyz.shape[0], 1), scale=xyz_std[2])
  xyz += np.hstack([x_delta, y_delta, z_delta])

  return Points(xyz=xyz, attr=cam_rgb_points.attr), labels


def random_drop(
  cam_rgb_points:Points, labels:dict, drop_prob:float=0.5, tier_prob:list=None):
  """ Randomly drops some points from the input points
  
  @param cam_rgb_points:  a Points object containing "xyz" and "attr".
  @param labels:          a dictionary containing the bounding box labels.
  @param drop_prob:       a float or list containing the probability of dropping 
                          points.
  @param tier_prob:       a list containing the probability of choosing the 
                          drop_prob.

  @return: a tuple of (Points, dict) containing the dropped points and labels.
  """
  if isinstance(drop_prob, list):                                               # choose drop probability according to prob distr. 
    drop_prob = np.random.choice(drop_prob, p=tier_prob)

  xyz   = cam_rgb_points.xyz
  mask  = np.random.uniform(size=xyz.shape[0]) > drop_prob                      # create mask for points to keep   

  if np.sum(mask) == 0:                                                         # catch case wehere all points would be dropped
    warnings.warn(
      "Warning: attempt to drop all points, falling back to all points", 
      UserWarning)
    mask = np.ones_like(mask)

  return Points(xyz=xyz[mask], attr=cam_rgb_points.attr[mask]), labels          # TODO: labels is redundant here


def random_global_drop(cam_rgb_points:Points, labels:dict, drop_std:float=0.25):
  """  Randomly drop points from the input points with a global drop probability
  
  @param cam_rgb_points: a Points object containing "xyz" and "attr".
  @param labels:         a dictionary containing the bounding box labels.
  @drop_std:             a standard deviation of drop prob. ditr.

  @return: a tuple of (Points, dict) containing the dropped points and labels.
  """
  drop_prob = np.abs(np.random.normal(scale=drop_std))                          # define "probability distribution"
  return random_drop(cam_rgb_points, labels, drop_prob=drop_prob)


def random_voxel_downsample(
  cam_rgb_points:Points, labels:dict, voxel_std:float=0.2,min_voxel:float=0.02, 
  max_voxel:float=0.8):
  """ Downnsamples the points by randomly choosing a voxel size

  @param cam_rgb_points: a Points object containing "xyz" and "attr".
  @param labels:         a dictionary containing the bounding box labels.
  @param voxel_std:      a standard deviation of the voxel size.
  @param min_voxel:      a minimum voxel size.
  @param max_voxel:      a maximum voxel size.

  @return: a tuple of (Points, dict) containing the downsampled points and labels.
  """

  voxel_size = np.abs(np.random.normal(scale=voxel_std))                        # define voxel size randomly          
  voxel_size = np.minimum(voxel_size, max_voxel)                                # voxel size needs to be within limit

  if voxel_size < min_voxel:  return cam_rgb_points, labels                     # voxels need to have min size

  downsampled_points = downsample_by_voxel(                                     # downsample points randomly
    cam_rgb_points,voxel_size, method='RANDOM',add_rnd3d=True)
  
  return downsampled_points, labels


def random_rotation_all(
  cam_rgb_points:Points, labels:dict, method_name:str='normal',
  yaw_std:float=0.3):
  """ Rotates the points and labels randomly
  
  @param cam_rgb_points: a Points object containing "xyz" and "attr".
  @param labels:         a dictionary containing the bounding box labels.
  @param method_name:    random smapling method.
  @param yaw_std:        standard deviation of the yaw angle.

  @return: a tuple of (Points, dict) containing the rotated points and labels.
  """

  xyz = cam_rgb_points.xyz

  if method_name == 'normal':                                                   # define random rotation angle
    delta_yaw = np.random.normal(scale=yaw_std)
  elif method_name == 'uniform':
    delta_yaw = np.random.uniform(low=-yaw_std, high=yaw_std)
  else:
    raise ValueError(f"Unknown method_name: {method_name}")

  R   = M_ROT(delta_yaw)                                                        # apply rotation to all points
  xyz = xyz.dot(np.transpose(R))

  for label in labels:                                                          # iterate over all bboxes
    if label['name'] != 'DontCare':
      tx = label['x3d']                                                         # get bbox centers
      ty = label['y3d']
      tz = label['z3d']
      xyz_center    = np.array([[tx, ty, tz]])
      xyz_center    = xyz_center.dot(np.transpose(R))                           # rotate bbxox
      label['x3d'], label['y3d'], label['z3d'] = xyz_center[0]                  # update label dictionary       
      label['yaw']  = label['yaw']+delta_yaw                                    

  return Points(xyz=xyz, attr=cam_rgb_points.attr), labels


def random_flip_all(cam_rgb_points:Points, labels:dict, flip_prob:float=0.5):
  """ Flips (or not) the points and labels randomly

  @param cam_rgb_points: a Points object containing "xyz" and "attr".
  @param labels:         a dictionary containing the bounding box labels.
  @param flip_prob:      probability of flipping the points.

  @return:  a tuple of (Points, dict) containing the flipped (or not) points and 
            labels.
  """
  xyz = cam_rgb_points.xyz                                                      # get points
  p   =  np.random.uniform()                                                    # flip points randomly                

  if p < flip_prob:                                                             # randomly flip along x-axis
    xyz[:,0] = -xyz[:,0]
    for label in labels:
      if label['name'] != 'DontCare':
        label['x3d'] = -label['x3d']              
        label['yaw'] = np.pi-label['yaw']

  return Points(xyz=xyz, attr=cam_rgb_points.attr), labels


def random_scale_all(
  cam_rgb_points:Points, labels:dict, method_name:str='normal',
  scale_std:float=0.05):
  """ Randomly scales (or not) the points and labels
  
  @paaram cam_rgb_points: a Points object containing "xyz" and "attr".
  @param labels:          a dictionary containing the bounding box labels.
  @param method_name:     random scaling method.
  @param scale_std:       standard deviation of the scaling factor.

  @return:  a tuple of (Points, dict) containing the scaled (or not) points and 
            labels.
  """

  xyz = cam_rgb_points.xyz

  if method_name == 'normal': scale = np.random.normal(scale=scale_std) + 1.0   # define scaling factor
  elif method_name == 'uniform':
    scale = np.random.uniform(low=-scale_std, high=scale_std) + 1
  else:
    raise ValueError(f"Unknown method_name: {method_name}")
  xyz *= scale

  for label in labels:                                                          # scale all bboxes
    if label['name'] != 'DontCare':
      label['x3d']    *= scale
      label['y3d']    *= scale
      label['z3d']    *= scale
      label['length'] *= scale
      label['width']  *= scale
      label['height'] *= scale

  return Points(xyz=xyz, attr=cam_rgb_points.attr), labels


def random_box_rotation(
  cam_rgb_points:Points, labels:dict, max_overlap_num_allowed:float=0.1,
  max_trails:int=100,method_name:str='normal',yaw_std:float=0.3,
  expend_factor:tuple=(1.0, 1.1, 1.1),augment_list:list=[
    'Car','Pedestrian','Cyclist','Van','Truck','Misc','Tram','Person_sitting',]
  ):
  """ Randomly rotates the bounding boxes 
  
  @param cam_rgb_points:  a Points object containing "xyz" and "attr".
  @param labels:          a dictionary containing the bounding box labels.
  @param max_overlap_num_allowed: 
                          maximum number of overlapping points.
  @param max_trails:      maximum number of trials.
  @param method_name:     random rotation method.
  @param yaw_std:         standard deviation of the yaw angle.
  @param expend_factor:   scaling factors for the box.
  @param augment_list:    list of objects to augment.
  """

  labels_no_dontcare = list(                                                    # filter out DontCare
    filter(lambda label: label['name'] != 'DontCare', labels))
  
  new_labels = []
  for i, label in enumerate(labels_no_dontcare):
    if label['name'] in augment_list:
      sucess = False
      for trial in range(max_trails):
        if method_name == 'normal': delta_yaw = np.random.normal(scale=yaw_std) # define random rotation angle
        else:
          if method_name == 'uniform':
            delta_yaw = np.random.uniform(low=-yaw_std,high=yaw_std)

        new_label = deepcopy(label)
        new_label['yaw'] = new_label['yaw']+delta_yaw                           # add noise to yaw angle
        mask      = sel_points_in_box3d(                                        # identify points in original bbox
          label, cam_rgb_points.xyz, expend_factor)                             
        more_mask = sel_points_in_box3d(new_label,                              # identify points in noise infected bbox
          cam_rgb_points.xyz[np.logical_not(mask)], expend_factor)
        
        if np.sum(more_mask) < max_overlap_num_allowed:                         # THIS CAN'T BE CORRECT !!
          mask  = sel_points_in_box3d(label, cam_rgb_points.xyz, expend_factor)
          points_xyz = cam_rgb_points.xyz[mask, :]
          tx    = label['x3d']
          ty    = label['y3d']
          tz    = label['z3d']
          points_xyz -= np.array([tx, ty, tz])                                  # move to origin    
          R     = M_ROT(delta_yaw)                                              # apply rotation
          points_xyz    = points_xyz.dot(np.transpose(R))
          points_xyz    = points_xyz+np.array([tx, ty, tz])                     # move back  
          xyz   = cam_rgb_points.xyz         
          xyz[mask, :]  = points_xyz
          new_labels.append(new_label)                                          # update boxes and label
          sucess = True
          break
      if not sucess:
        warnings.warn('Warning: failed to augment by rotation', UserWarning)
        new_labels.append(label)
    else:
      new_labels.append(label)

  assert len(new_labels) == len(labels_no_dontcare)
  new_labels.extend([l for l in labels if l['name'] == 'DontCare'])
  assert len(new_labels) == len(labels)

  return Points(xyz=xyz, attr=cam_rgb_points.attr), new_labels


def random_box_global_rotation(
  cam_rgb_points:Points, labels:dict,max_overlap_num_allowed:float=0.1,
  max_trails:int=100,method_name:str='normal', yaw_std:float=0.3, 
  expend_factor:tuple=(1.1, 1.1, 1.1),augment_list:list=[
    'Car','Pedestrian','Cyclist','Van','Truck','Misc','Tram','Person_sitting',]
  ):
  """ Randomly rotates the bounding boxes globally 
  
  @param cam_rgb_points:  a Points object containing "xyz" and "attr".
  @param labels:          a dictionary containing the bounding box labels.
  @param max_overlap_num_allowed:
                          maximum number of overlapping points.
  @param max_trails:      maximum number of trials.
  @param method_name:     random rotation method.
  @param yaw_std:         standard deviation of the yaw angle.
  @param expend_factor:   scaling factors for the box.
  @param augment_list:    list of objects

  @return: a tuple of (Points, dict) containing the rotated points and labels.
  """
  xyz   = cam_rgb_points.xyz
  attr  = cam_rgb_points.attr
  labels_no_dontcare = list(
    filter(lambda label: label['name'] != 'DontCare', labels))                  # filtering out DontCare
  
  new_labels = []
  for i, label in enumerate(labels_no_dontcare):                                # check existing overlap
    if label['name'] in augment_list:
      trial   = 0
      sucess  = False

      for trial in range(max_trails):
        if method_name == 'normal':                                             # define random rotation angle
          delta_yaw = np.random.normal(scale=yaw_std)
        else:
          if method_name == 'uniform':
            delta_yaw = np.random.uniform(low=-yaw_std, high=yaw_std)

          new_label = deepcopy(label)
          new_label['yaw'] = new_label['yaw']+delta_yaw
          tx  = new_label['x3d']
          ty  = new_label['y3d']
          tz  = new_label['z3d']
          R   = M_ROT(delta_yaw)

          new_label['x3d'],new_label['y3d'],new_label['z3d'] = \
            np.array([tx, ty, tz]).dot(np.transpose(R))
          
          mask      = sel_points_in_box3d(label, xyz, expend_factor)            # check if the new box includes more points
          new_mask  = sel_points_in_box3d(new_label, xyz, expend_factor)
          more_mask = np.logical_and(new_mask, np.logical_not(mask))

          if np.sum(more_mask) < max_overlap_num_allowed:                       # TODO: THIS CAN'T BE CORRECT !!
            points_xyz = xyz[mask, :]                                           # valid new box, start rotation
            points_xyz = points_xyz.dot(np.transpose(R))
            # points_xyz = points_xyz+np.array([tx, ty, tz])
            xyz[mask, :] = points_xyz
            xyz     = xyz[np.logical_not(more_mask)]
            attr    = attr[np.logical_not(more_mask)]
            new_labels.append(new_label)                                        # update boxes and label
            sucess  = True
            break

      if not sucess:
        warnings.warn('Warning: fail to augment by rotation', UserWarning)
        new_labels.append(label)
    else:
        new_labels.append(label)

  assert len(new_labels) == len(labels_no_dontcare)
  new_labels.extend([l for l in labels if l['name'] == 'DontCare'])
  assert len(new_labels) == len(labels)

  return Points(xyz=xyz, attr=attr), new_labels


def random_box_shift(
  cam_rgb_points:Points, labels:dict, max_overlap_num_allowed:float=0.1,
  max_overlap_rate:float=None, max_trails:int = 100, appr_factor:int=100,
  method_name:str='normal', xyz_std:tuple=(1,0,1), 
  expend_factor:tuple=(1.0, 1.1, 1.1),augment_list:list=[
    'Car','Pedestrian','Cyclist','Van','Truck','Misc','Tram','Person_sitting',],
  shuffle:bool=False):
  """ Randomly shifts the bounding boxes 
  
  @param cam_rgb_points:  a Points object containing "xyz" and "attr".
  @param labels:          a dictionary containing the bounding box labels.
  @param max_overlap_num_allowed:
                          maximum number of overlapping points.       
  @param max_overlap_rate:
                          maximum overlap rate.     
  @param max_trails:      maximum number of trials.
  @param appr_factor:     approximation factor.
  @param method_name:     random shifting method.
  @param xyz_std:         standard deviation of the shift.
  @param expend_factor:   scaling factors for the box.
  @param augment_list:    list of objects to augment.
  @param shuffle:         shuffle the labels.

  @return: a tuple of (Points, dict) containing the shifted points and labels.
  """
  xyz = cam_rgb_points.xyz
  labels_no_dontcare = list(
    filter(lambda label: label['name'] != 'DontCare', labels))                  # filtering out DontCare
  
  if shuffle: random.shuffle(labels_no_dontcare)
  
  new_labels = []
  label_boxes_corners = None

  for i, label in enumerate(labels_no_dontcare):                                # check existing overlap                
    if label['name'] in augment_list:
      trial   = 0
      sucess  = False

      for trial in range(max_trails):
          
        if method_name == 'normal':                                             # random rotation method
          delta_x, delta_y, delta_z = np.random.normal(scale=xyz_std)
        else:
          if method_name == 'uniform':
            delta_x, delta_y, delta_z = np.random.uniform(
              ow=-xyz_std, high=xyz_std)
            
        new_label = deepcopy(label)
        new_label['x3d'] = new_label['x3d']+delta_x
        new_label['y3d'] = new_label['y3d']+delta_y
        new_label['z3d'] = new_label['z3d']+delta_z
        
        below_overlap = True
        mask      = sel_points_in_box3d(label, xyz, expend_factor)              # check if the new box includes more points
        more_mask = sel_points_in_box3d(new_label,
            xyz[np.logical_not(mask)], expend_factor)
        below_overlap *= np.sum(more_mask) < max_overlap_num_allowed

        if max_overlap_rate is not None:
          new_boxes = np.array([                                                # get labels in array form
            [new_label['x3d'],
              new_label['y3d'],
              new_label['z3d'],
              new_label['length'],
              new_label['height'],
              new_label['width'],
              new_label['yaw']]
              ])
          new_boxes_corners = np.int32(                                         # convert label array to bounding boxes
            appr_factor*boxes_3d_to_corners(new_boxes))
          label_boxes = np.array([
            [l['x3d'], l['y3d'], l['z3d'],
            l['length'], l['height'], l['width'], l['yaw']]
            for l in new_labels])
          label_boxes_corners = np.int32(                                       # convert original labels to bounding boxes
            appr_factor*boxes_3d_to_corners(label_boxes))
          below_overlap_rate = np.all(overlapped_boxes_3d_fast_poly(            # check overlap rate
            new_boxes_corners[0],
            label_boxes_corners) < max_overlap_rate)
          below_overlap *= below_overlap_rate

        if below_overlap:                                                       # valid new box, start rotation                  
          mask = sel_points_in_box3d(label, xyz, expend_factor)                 
          points_xyz    = xyz[mask, :]
          points_xyz    = points_xyz+np.array([delta_x, delta_y, delta_z])
          xyz[mask, :]  = points_xyz
          new_labels.append(new_label)                                          # update boxes and label
          sucess = True
          break
      if not sucess:
        warnings.warn('Warning: fail to augment by shifting', UserWarning)
        new_labels.append(label)
    else:
      new_labels.append(label)
  assert len(new_labels) == len(labels_no_dontcare)
  new_labels.extend([l for l in labels if l['name'] == 'DontCare'])
  assert len(new_labels) == len(labels)

  return Points(xyz=xyz, attr=cam_rgb_points.attr), new_labels


def split_background(points:Points, labels:dict, expend_factor:tuple,keep_list:list,num_object:int=-1):     
  mask  = np.zeros(points.xyz.shape[0], dtype=np.bool)

  labels_no_dontcare = list(
    filter(lambda label: label['name'] in keep_list, labels))                   # filtering out DontCare

  selected_labels = []
  if num_object > 0:
    sample_idx = np.random.choice(len(labels_no_dontcare), num_object)
    for i in sample_idx:  selected_labels.append(labels_no_dontcare[i])
  else:
    selected_labels = labels_no_dontcare
  selected_labels = deepcopy(labels_no_dontcare)

  for label in selected_labels:                                                 # select all points belonging to any bbox 
    mask += sel_points_in_box3d(label, points.xyz, expend_factor)

  #assert mask.any()
  if not mask.any():  mask[0] = True                                            # keep one point in foreground

  background_xyz    = points.xyz[np.logical_not(mask)]                          # split into background and objects
  background_attr   = points.attr[np.logical_not(mask)]
  front_xyz   = points.xyz[mask]
  front_attr  = points.attr[mask]

  points_bg = Points(xyz=background_xyz, attr=background_attr)
  points_fg = Points(xyz=front_xyz, attr=front_attr)

  return points_bg, points_fg, labels_no_dontcare                                                         


def dilute_background(cam_rgb_points, labels, dilute_voxel_base=0.4,
  expend_factor=(4.0, 4.0, 4.0),
  keep_list=['Car','Pedestrian','Cyclist','Van','Truck','Misc','Person_sitting'],
  ):
  """ Dilutes the background points
  
  @param cam_rgb_points:  a Points object containing "xyz" and "attr".
  @param labels:          a dictionary containing the bounding box labels.
  @param dilute_voxel_base:
                          base voxel size for dilution.
  @param expend_factor:   scaling factors for the box.
  """
  points_bg, points_fg, labels_no_dontcare = split_background(
    cam_rgb_points, labels, expend_factor, keep_list)

  diluted_background_points = downsample_by_voxel(                              # dilute background points 
    points_bg, dilute_voxel_base, method='RANDOM',add_rnd3d=True)

  return Points(                                                                # concatenate front and diluted background points
    xyz=np.concatenate([points_fg.xyz, diluted_background_points.xyz], axis=0),
    attr=np.concatenate([
      points_fg.attr,diluted_background_points.attr], axis=0)
    ), labels_no_dontcare


def remove_background(cam_rgb_points, labels, expend_factor=(4.0, 4.0, 4.0),
  keep_list=['Car','Pedestrian','Cyclist','Van','Truck','Misc','Person_sitting',],
  num_object=-1):
  """ Removes the background points
  
  @param cam_rgb_points:  a Points object containing "xyz" and "attr".
  @param labels:          a dictionary containing the bounding box labels.
  @param expend_factor:   scaling factors for the box.
  @param keep_list:       list of objects to keep.
  @param num_object:      number of objects to keep.
  """
  points_bg, points_fg, labels_no_dontcare = split_background(
    cam_rgb_points, labels, expend_factor, keep_list,num_object)

  return points_fg, labels_no_dontcare


def random_transition(cam_rgb_points, labels, xyz_std=(0.1, 0.1, 0.1)):
  xyz     = cam_rgb_points.xyz
  x_delta = np.random.normal(scale=xyz_std[0])
  y_delta = np.random.normal(scale=xyz_std[1])
  z_delta = np.random.normal(scale=xyz_std[2])
  xyz += np.hstack([x_delta, y_delta, z_delta])

  for label in labels:                                                          # translate bboxes equally randomly
    label['x3d'] += x_delta
    label['y3d'] += y_delta
    label['z3d'] += z_delta

  return Points(xyz=xyz, attr=cam_rgb_points.attr), labels


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
  
  Parameters 
  __________
  aug_configs dict: 
    a list of dictionaries containing the augmentation configurations.

  Yields:
  ________
  a function that applies augmentation according to configurations.
  """

  empty = lambda cam_rgb_points, labels: (cam_rgb_points, labels)
  if len(aug_configs)==0: 
    yield empty

  def multiple_aug(cam_rgb_points, labels):
    for aug_config in aug_configs:
      aug_method = AUG_METHOD_MAP[aug_config['method_name']]
      cam_rgb_points, labels = aug_method(
        cam_rgb_points, labels, **aug_config['method_kwargs'])
    yield cam_rgb_points, labels
  
  return multiple_aug
