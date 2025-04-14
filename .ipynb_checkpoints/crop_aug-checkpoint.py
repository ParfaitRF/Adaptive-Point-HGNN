"""This file implement augmentation by cropping and parsing ground truth boxes"""
import warnings
import json
from collections import defaultdict

import numpy as np
import open3d
from copy import deepcopy
from tqdm import tqdm

from data.kitti_dataset import KittiDataset
from globals import M_ROT
from data.transformations import boxes_3d_to_corners
from utils.nms import overlapped_boxes_3d_fast_poly
from data.utils import Points,sel_points_in_box2d,sel_points_in_box3d

#from models import preprocess

def save_cropped_boxes(
        dataset:KittiDataset, filename:str, expand_factor:tuple=[1.1, 1.1, 1.1],
        minimum_points:int=10,backlist:list=[]):
  """ creates a json file containing filtered labels and points within them

  @param dataset:         KittiDataset object
  @param filename:        str, the file to save the cropped boxes
  @param expand_factor:   list of float, the expand factor for cropping
  @param minimum_points:  int, the minimum number of points in a cropped box
  @param backlist:        list of str, the list of object names to be ignored
  """

  cropped_labels      = defaultdict(list)
  cropped_cam_points  = defaultdict(list)

  for frame_idx in tqdm(range(dataset.num_files)):
    labels      = dataset.get_label(frame_idx)
    cam_points  = dataset.get_cam_points_in_image_with_rgb(frame_idx)

    for label in labels:                                                        # iterate all labels                
      if label['name'] != "DontCare":                                           # check if we care about label
        if label['name'] not in backlist:                                       # TODO: blacklist defintion                        
          mask = dataset.sel_points_in_box3d(label, cam_points.xyz,expand_factor)   # get mask of points in box

          if np.sum(mask) > minimum_points:                                     # sufficiently many points in bbox
            cropped_labels[label['name']].append(label)
            cropped_cam_points[label['name']].append(
              [cam_points.xyz[mask].tolist(),cam_points.attr[mask].tolist()])

  with open(filename, 'w') as outfile:                                          # save cropped data to file
    json.dump((cropped_labels,cropped_cam_points), outfile)


def load_cropped_boxes(filename:str):
  """ load cropped boxes from json file

  @param filename:  str, the file to load
  @return:          dict, dict, the labels and points in the cropped boxes
  """
  with open(filename, 'r') as infile:                                           # load cropped data from file            
    cropped_labels, cropped_cam_points = json.load(infile)

  for key in cropped_cam_points:
    print("Loaded %d %s" % (len(cropped_cam_points[key]), key))

    for i, cam_points in enumerate(cropped_cam_points[key]):                    # convert json to dictionary
      cropped_cam_points[key][i] = Points(xyz=np.array(cam_points[0]),
                                          attr=np.array(cam_points[1]))
        
  return cropped_labels, cropped_cam_points


def vis_cropped_boxes(cropped_labels:dict, cropped_cam_points:dict, 
                      dataset:KittiDataset,object_class:list='Pedestrian'):
  """ Visualizes cropped boxes

  @param cropped_labels:      dict, the labels in the cropped boxes
  @param cropped_cam_points:  dict, the points in the cropped boxes
  @param dataset:             KittiDataset object 
  """

  for key in cropped_cam_points:                                                # iteratte over unique object names
    if key == object_class:                                                     # visualize only selected
      for i, cam_points in enumerate(cropped_cam_points[key]):
        label = cropped_labels[key][i]
        print(label['name'])
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(cam_points.xyz)
        pcd.colors = open3d.utility.Vector3dVector(cam_points.attr[:, 1:])

        def custom_draw_geometry_load_option(geometry_list):
          vis = open3d.visualization.Visualizer()
          vis.create_window()
          for geometry in geometry_list:  vis.add_geometry(geometry)
          ctr = vis.get_view_control()
          ctr.rotate(0.0, 3141.0, 0)
          vis.run()
          vis.destroy_window()

        custom_draw_geometry_load_option(                                       # draw points and boxes
          [pcd] + dataset.get_open3D_box(label))


def parser_without_collision(
  cam_rgb_points:Points,labels:dict,sample_cam_points:Points,sample_labels:dict,
  overlap_mode:str='box',auto_box_height:bool=False,max_overlap_rate:float=0.01,
  appr_factor:int=100,max_overlap_num_allowed:int=1, max_trails:int=1, 
  method_name:str='normal',yaw_std:float=0.3, 
  expand_factor:tuple=(1.1, 1.1, 1.1),must_have_ground:bool=False):
  """ Parse cropped boxes to a frame without collision

  @param cam_rgb_points:	    Existing scene point cloud, a Points object (xyz coordinates + attributes).
  @param labels:  	          List of dictionaries describing existing objects (3D boxes) already in the scene.
  @param sample_cam_points:	  Point clouds associated with new objects to be inserted.
  @param sample_labels: 	    Labels (box definitions) for the new objects to be inserted.
  @param overlap_mode:	      How overlap is checked (box, point, or both).
  @param auto_box_height:	    Whether to auto-adjust box height to ground level.
  @param max_overlap_rate>	  Maximum allowed overlap between boxes (for box mode).
  @param appr_factor:	        Scaling factor applied to boxes when checking overlap.
  @param max_overlap_num_allowed:	
                              Maximum allowed number of overlapping points.
  @param max_trails:	        Max retries per box to find a valid, non-overlapping position.
  @param method_name:	        How rotation is sampled (normal or uniform).
  @param yaw_std:	            Standard deviation for random yaw sampling.
  @param expand_factor:	      Expansion factor for boxes (for height adjustment and overlap checks).
  @param must_have_ground:	  If True, box is only valid if some ground points exist under it.
  
  """
  
  xyz   = cam_rgb_points.xyz
  attr  = cam_rgb_points.attr

  if overlap_mode in ['box','box_and_point']:
    label_boxes = np.array(
      [[l['x3d'], l['y3d'], l['z3d'], l['length'],
        l['height'], l['width'], l['yaw']] for l in labels ])
    label_boxes_corners = np.int32(                                             # get label boxes corners
      appr_factor*boxes_3d_to_corners(label_boxes))
    
  for i, label in enumerate(sample_labels):                                     # iterate over all labels
    sucess  = False
    for trial in range(max_trails):
      if method_name == 'normal':                                               # get random yaw               
        delta_yaw = np.random.normal(scale=yaw_std)
      elif method_name == 'uniform':
        delta_yaw = np.random.uniform(low=-yaw_std, high=yaw_std)

      new_label = deepcopy(label)
      R   = M_ROT(delta_yaw)                                                    # get rotation matrix
      tx  = new_label['x3d']                                                    # get bbox center
      ty  = new_label['y3d']
      tz  = new_label['z3d']

      xyz_center = np.array([[tx, ty, tz]])                                     # bbox center    
      xyz_center = xyz_center.dot(np.transpose(R))

      new_label['x3d'], new_label['y3d'], new_label['z3d'] = xyz_center[0]      # apply random rotation to center
      new_label['yaw'] = new_label['yaw']+delta_yaw

      if auto_box_height:                                                       # TODO: define functionalitys
        original_height = new_label['height']
        mask_2d         = sel_points_in_box2d(
          new_label, cam_rgb_points, expand_factor)
        
        if np.sum(mask_2d):                                                     # check if any points in bbox
          ground_height = np.amax(cam_rgb_points.xyz[mask_2d][:,1])             # height from ground as argmax of points y value
          y3d_adjust    = ground_height - new_label['y3d']                      # tranlation value for y translation
        else:
          if must_have_ground: continue
          y3d_adjust = 0

        # if np.abs(y3d_adjust) > 1:
        #     y3d_adjust = 0
        new_label['y3d'] += y3d_adjust                                          # translate object to the ground
        new_label['height'] = original_height                                   # REDUNDANT ?? (line 143)

      mask = sel_points_in_box3d(new_label, cam_rgb_points.xyz, expand_factor)
      # check if the new box includes more points than before

      if overlap_mode == 'box':
        new_boxes = np.array([
            [new_label['x3d'],
              new_label['y3d'],
              new_label['z3d'],
              new_label['length'],
              new_label['height'],
              new_label['width'],
              new_label['yaw']]
              ])
        new_boxes_corners = np.int32(                                           # get bbox vertices
          appr_factor*boxes_3d_to_corners(new_boxes))
        below_overlap     = np.all(overlapped_boxes_3d_fast_poly(               # compute overlap between first new bbox and all label boxes
          new_boxes_corners[0],
          label_boxes_corners) < max_overlap_rate)
      elif overlap_mode == 'point':
        below_overlap = np.sum(mask) < max_overlap_num_allowed
      elif overlap_mode == 'box_and_point':
        new_boxes = np.array([
            [new_label['x3d'],
              new_label['y3d'],
              new_label['z3d'],
              new_label['length'],
              new_label['height'],
              new_label['width'],
              new_label['yaw']]
              ])
        new_boxes_corners = np.int32(appr_factor*boxes_3d_to_corners(new_boxes))
        below_overlap     = np.all(
          overlapped_boxes_3d_fast_poly(new_boxes_corners[0],
          label_boxes_corners) < max_overlap_rate)
        below_overlap     = np.logical_and(below_overlap,
            (np.sum(mask) < max_overlap_num_allowed))
      else: raise Exception('Unknown overlap mode!')
        
      if below_overlap:
        points_xyz  = sample_cam_points[i].xyz
        points_xyz  = points_xyz.dot(np.transpose(R))
        points_attr = sample_cam_points[i].attr
        
        if auto_box_height: points_xyz[:,1] = points_xyz[:,1] + y3d_adjust

        xyz   = xyz[np.logical_not(mask)]
        xyz   = np.concatenate([points_xyz, xyz], axis=0)
        attr  = attr[np.logical_not(mask)]
        attr  = np.concatenate([points_attr, attr], axis=0)
        # update boxes and label
        labels.append(new_label)

        if overlap_mode in ['box','box_and_point']:
          label_boxes_corners = np.append(label_boxes_corners,
                                          new_boxes_corners,axis=0)
        sucess = True
        break
    if not sucess:
      #if not sucess: keep the old label
      warnings.warn("Failed to parse cropped box", UserWarning)
  return Points(xyz=xyz, attr=attr), labels


class CropAugSampler():
  """ A class to sample from cropped objects and parse it to a frame """
  def __init__(self, crop_filename:str):
    """ Initialize the class with a crop file 
    @param crop_filename:  str, the file containing the cropped boxes
    """

    self._cropped_labels, self._cropped_cam_points = load_cropped_boxes(\
        crop_filename)
    

  def crop_aug(self, cam_rgb_points:Points, labels:dict,
               sample_rate:dict={"Car":1, "Pedestrian":1, "Cyclist":1},
               parser_kwargs:dict={}):
    """ Crop and parse the frame with the cropped boxes
    @param cam_rgb_points:  Points, the point cloud in the frame
    """
    sample_labels     = []
    sample_cam_points = []

    for key in sample_rate:
      sample_indices = np.random.choice(len(self._cropped_labels[key]),
          size=sample_rate[key], replace=False)
      sample_labels.extend(
          deepcopy([self._cropped_labels[key][idx]
              for idx in sample_indices]))
      sample_cam_points.extend(
          deepcopy([self._cropped_cam_points[key][idx]
              for idx in sample_indices]))
      
    return parser_without_collision(cam_rgb_points, labels,sample_cam_points, 
                                    sample_labels,**parser_kwargs)


def vis_crop_aug_sampler(crop_filename, dataset):
  sampler = CropAugSampler(crop_filename)
  for frame_idx in range(10):
      labels = dataset.get_label(frame_idx)
      cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(frame_idx)
      cam_rgb_points, labels = sampler.crop_aug(cam_rgb_points, labels,
          sample_rate={"Car":2, "Pedestrian":10, "Cyclist":10},
          parser_kwargs={
              'max_overlap_num_allowed': 50,
              'max_trails':100,
              'method_name':'normal',
              'yaw_std':np.pi/16,
              'expand_factor':(1.1, 1.1, 1.1),
              'auto_box_height': True,
              'overlap_mode':'box_and_point',
              'max_overlap_rate': 1e-6,
              'appr_factor': 100,
              'must_have_ground': True,
              })
      aug_configs = [
          {'method_name': 'random_box_global_rotation',
            'method_kwargs': { 'max_overlap_num_allowed':100,
                              'max_trails': 100,
                              'appr_factor':100,
                              'method_name':'normal',
                              'yaw_std':np.pi/8,
                              'expend_factor':(1.1, 1.1, 1.1)
                              }
          }
      ]
      aug_fn = preprocess.get_data_aug(aug_configs)
      cam_rgb_points, labels = aug_fn(cam_rgb_points, labels)
      dataset.vis_points(cam_rgb_points, labels, expend_factor=(1.1, 1.1,1.1))


    # # Example of usage
    # print('generate training split: ')
    # kitti_train = KittiDataset(
    #     '../dataset/kitti/image/training/image_2',
    #     '../dataset/kitti/velodyne/training/velodyne/',
    #     '../dataset/kitti/calib/training/calib/',
    #     '../dataset/kitti/labels/training/label_2/',
    #     '../dataset/kitti/3DOP_splits/train.txt',)
    # save_cropped_boxes(kitti_train, "../dataset/kitti/cropped/car_person_cyclist_train.json",
    #     expand_factor = (1.1, 1.1, 1.1), minimum_points=10,
    #     backlist=['Van', 'Truck', 'Misc', 'Tram', 'Person_sitting'])
    # print("generate val split: ")
    # kitti_val = KittiDataset(
    #     '../dataset/kitti/image/training/image_2',
    #     '../dataset/kitti/velodyne/training/velodyne/',
    #     '../dataset/kitti/calib/training/calib/',
    #     '../dataset/kitti/labels/training/label_2/',
    #     '../dataset/kitti/3DOP_splits/val.txt',)
    # save_cropped_boxes(kitti_val, "../dataset/kitti/cropped/car_person_cyclist_val.json",
    #     expand_factor = (1.1, 1.1, 1.1), minimum_points=10,
    #     backlist=['Van', 'Truck', 'Misc', 'Tram', 'Person_sitting'])
    # print("generate trainval: ")
    # kitti_trainval = KittiDataset(
    #     '../dataset/kitti/image/training/image_2',
    #     '../dataset/kitti/velodyne/training/velodyne/',
    #     '../dataset/kitti/calib/training/calib/',
    #     '../dataset/kitti/labels/training/label_2/',
    #     '../dataset/kitti/3DOP_splits/trainval.txt',)
    # save_cropped_boxes(kitti_trainval, "../dataset/kitti/cropped/car_person_cyclist_trainval.json",
    #     expand_factor = (1.1, 1.1, 1.1), minimum_points=10,
    #     backlist=['Van', 'Truck', 'Misc', 'Tram', 'Person_sitting'])
    # cropped_labels, cropped_cam_points = load_cropped_boxes(
    #     "../dataset/kitti/cropped/car_person_cyclist_train.json")
    # vis_cropped_boxes(cropped_labels, cropped_cam_points, kitti_train)
    # cropped_labels, cropped_cam_points = load_cropped_boxes(
    #     "../dataset/kitti/cropped/car_person_cyclist_val.json")
    # vis_cropped_boxes(cropped_labels, cropped_cam_points, kitti_val)
    # cropped_labels, cropped_cam_points = load_cropped_boxes(
    #     "../dataset/kitti/cropped/car_person_cyclist_trainval.json")
    # vis_cropped_boxes(cropped_labels, cropped_cam_points, kitti_trainval)
    # vis_crop_aug_sampler("../dataset/kitti/cropped/car_person_cyclist_val.json", kitti_val)
