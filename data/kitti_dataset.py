import os
from os.path import isfile, join
from copy import deepcopy
import json
import numpy as np
import cv2
import open3d as o3d
from collections import defaultdict
from tqdm import tqdm
import json
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor,as_completed


from globals import (
  M_ROT,BOX_OFFSET,Points,
  IMG_WIDTH, IMG_HEIGHT,                                                        # image dimensions
  OBJECT_HEIGHT_THRESHOLDS,TRUNCATION_THRESHOLDS,OCCULUSION_THRESHOLDS,         # thresholds
  OCCLUSION_COLORS,COLOR_MAP,LABEL_MAP                                       # colors
)
from .transformations import (
  boxes_3d_to_corners,box3d_to_cam_points,cam_points_to_image,velo_to_cam,
)
from .utils import (
  sel_points_in_box3d,sel_points_in_box2d,downsample_by_voxel
)
from .preprocess import get_data_aug
from util.nms import overlapped_boxes_3d_fast_poly

from models.graph_gen import (
  multi_layer_downsampling,
  gen_disjointed_rnn_local_graph_v3,
  gen_multi_level_local_graph_v3
)

class KittiDataset(object):
  """ A class for interactions with the KITTI dataset """
  # ===========================================================================#
  # ============================= INITIALIZATION ==============================#
  # ===========================================================================#

  def __init__(self,is_training:bool=True, is_raw:bool=False, difficulty:int=0,
               num_classes:int=8):
    """ Constructor
    @param is_training: determines if datasset is training set
    @param is_raw:      determines if dataset is raw dataset
    @param difficulty:  determines the difficulty of the dataset
    @param num_classes: number of classes in the dataset
    """

    self._is_training   = is_training
    self._is_raw        = is_raw
    self.num_classes    = num_classes
    self.difficulty     = difficulty
    # initialize input variables
    data_root_dir = os.path.dirname(os.path.abspath(__file__))
    kitti_root_dir = join(data_root_dir,'kitti')
    subfolder     = 'training' if is_training else 'testing'

    self._calib_dir = join(kitti_root_dir,subfolder,'calib')
    self._image_dir = join(kitti_root_dir,subfolder,'image_2')
    self._point_dir = join(kitti_root_dir,subfolder,'velodyne')
    self._crop_file = join(data_root_dir,'cropped_boxes\\cropped.json')

    if is_training: 
      self._label_dir = join(kitti_root_dir,subfolder,'label_2')

    self._file_list = self._get_file_list(self._image_dir)
    self._max_image_height  = IMG_HEIGHT
    self._max_image_width   = IMG_WIDTH

    self._verify_file_list()
    self.init_stats()
    self._cropped_labels, self._cropped_cam_points = self.load_cropped_boxes()

  # ===========================================================================#
  # =========================== STRING REPRESENTATION =========================#
  # ===========================================================================#

  def __str__(self):
    """ Generate a string summary of the dataset """

    summary_string = ('\nDataset Summary:\n'
      + '* Paths{\n'
      + '\timage_dir=%s\n' % self._image_dir
      + '\tpoint_dir=%s\n' % self._point_dir
      + '\tcalib_dir=%s\n' % self._calib_dir
      + '\tlabel_dir=%s\n' % self._label_dir
      + '}\n\n'
      + '* Total number of sampels: %d {\n' % self.num_files)
    statics = self.get_statics()

    return summary_string + statics
  
  
  def process_frame_stats(self,frame_idx):
    stats = {
      'x': defaultdict(list), 'y': defaultdict(list), 'z': defaultdict(list),
      'w': defaultdict(list), 'h': defaultdict(list), 'l': defaultdict(list),
      'view_angle': defaultdict(list), 'yaw': defaultdict(list),
      'trunc_rates': [], 'no_trunc_rates': [], 'img_height': 0, 'img_width': 0,
      'occlusion': defaultdict(list), 'truncation': defaultdict(list),
      'height': defaultdict(list)
    }

    labels  = self.get_label(frame_idx)
    calib   = self.get_calib(frame_idx)
    image   = self.get_image(frame_idx)
    stats['img_height']   = image.shape[0]
    stats['img_width']    = image.shape[1]

    for label in labels:                                                        # get all label information in frame
      if label['ymin'] > 0:
        object_name = label['name']  
        if label['name'] == 'Car':
          if label['ymax'] - label['ymin'] <= OBJECT_HEIGHT_THRESHOLDS[self.difficulty]:                
            object_name = 'ignored_by_height'
          elif label['truncation'] > TRUNCATION_THRESHOLDS[self.difficulty]:                        # too much truncation
            object_name = 'ignored_by_truncation'
          elif label['occlusion'] > OCCULUSION_THRESHOLDS[self.difficulty]:     # check occlusion rate
            object_name = 'ignored_by_occlusion'
      else: continue

      stats['w'][object_name].append(label['width'])                            # object width
      stats['h'][object_name].append(label['height'])                           # object height
      stats['l'][object_name].append(label['length'])                           # object length
      stats['x'][object_name].append(label['x3d'])                              # x coordinate
      stats['y'][object_name].append(label['y3d'])                             # y coordinate
      stats['z'][object_name].append(label['z3d'])                              # z coordinate
      stats['view_angle'][object_name].append(                                  # compute view angle  
        np.arctan(label['x3d']/label['z3d']))
      stats['yaw'][object_name].append(label['yaw'])

      detection_boxes_3d = np.array(                                            # get label 3D boxes
        [[label['x3d'], label['y3d'], label['z3d'], label['length'], 
          label['height'], label['width'], label['yaw']]])
      detection_boxes_3d_corners = boxes_3d_to_corners(detection_boxes_3d)      # translate from origin
      corners_cam_points = Points(                                              # convert to Points object  
        xyz=detection_boxes_3d_corners[0], attr=None)
      corners_img_points = cam_points_to_image(corners_cam_points, calib)       # convert velodyne into image points
      corners_xy  = corners_img_points.xyz[:, :2]                               # get x and y coordinates         
      xmin, ymin  = np.amin(corners_xy, axis=0)                                 # get min and max coordinates
      xmax, ymax  = np.amax(corners_xy, axis=0)
      clip_xmin   = max(xmin, 0.0)                                              # define clip off points
      clip_ymin   = max(ymin, 0.0)
      clip_xmax   = min(xmax, IMG_WIDTH)
      clip_ymax   = min(ymax, IMG_HEIGHT)
      truncation_rate = 1.0 - \
        (clip_ymax - clip_ymin)*(clip_xmax - clip_xmin)/\
          ((ymax - ymin)*(xmax - xmin))                                         # compute truncation rate
      
      if label['truncation'] > TRUNCATION_THRESHOLDS[self.difficulty]:          # check truncation rate
        stats['trunc_rates'].append(truncation_rate)
      else:
        stats['no_trunc_rates'].append(truncation_rate)
      
    return stats
  

  def init_stats(self):
    """ compute stats once for all objects in the dataset """

    # coordinates lists
    self.x_dict = defaultdict(list)
    self.y_dict = defaultdict(list)
    self.z_dict = defaultdict(list)
    # bounding box dimensions list
    self.h_dict = defaultdict(list)
    self.w_dict = defaultdict(list)
    self.l_dict = defaultdict(list)
    self.view_angle_dict = defaultdict(list) # view angle list
    self.yaw_dict        = defaultdict(list) # yaw list

    self.truncation_rates     = []
    self.no_truncation_rates  = []
    self.image_heights        = []
    self.image_widths         = []

    # get all relevant values in easy format
    with ThreadPoolExecutor() as executor:
      results = list(executor.map(self.process_frame_stats, range(self.num_files)))
    
    # Merge results
    for stats in results:
      for k, v in stats['x'].items():
        self.x_dict[k].extend(v)

      for k, v in stats['y'].items():
        self.y_dict[k].extend(v)

      for k, v in stats['z'].items():
        self.z_dict[k].extend(v)

      for k, v in stats['w'].items():
        self.w_dict[k].extend(v)

      for k, v in stats['h'].items():
        self.h_dict[k].extend(v)

      for k, v in stats['l'].items():
        self.l_dict[k].extend(v)

      for k, v in stats['view_angle'].items():
        self.view_angle_dict[k].extend(v)

      for k, v in stats['yaw'].items():
        self.yaw_dict[k].extend(v)


      self.truncation_rates.extend(stats['trunc_rates'])
      self.no_truncation_rates.extend(stats['no_trunc_rates'])
      self.image_heights.append(stats['img_height'])
      self.image_widths.append(stats['img_width'])


  def get_statics(self):
    """ Get statistics of objects in the dataset """

    statistics = ""
    for object_name in self.h_dict:
      # print(object_name+
      #   "l="+str(np.histogram(l_dict[object_name], 10, density=True))+'\n')
      
      if len(self.h_dict[object_name]) == 0: continue

      statistics += ('\t* ' + str(object_name) + 's {\n' 
        + '\t\t# objects= ' + str(len(self.h_dict[object_name])) + ";\n"
        + "\t\tmh= " + str(np.min(self.h_dict[object_name])) + " "
                + str(np.median(self.h_dict[object_name])) + " "
                + str(np.max(self.h_dict[object_name])) + ";\n"
        + "\t\tmw= " + str(np.min(self.w_dict[object_name])) + " "
                + str(np.median(self.w_dict[object_name])) + " "
                + str(np.max(self.w_dict[object_name])) + ";\n"
        + "\t\tml= " + str(np.min(self.l_dict[object_name])) + " "
                + str(np.median(self.l_dict[object_name])) + " "
                + str(np.max(self.l_dict[object_name])) + ";\n"
        + "\t\tmx= " + str(np.min(self.x_dict[object_name])) + " "
                + str(np.median(self.x_dict[object_name])) + " "
                + str(np.max(self.x_dict[object_name])) + ";\n"
        + "\t\tmy= " + str(np.min(self.y_dict[object_name])) + " "
                + str(np.median(self.y_dict[object_name])) + " "
                + str(np.max(self.y_dict[object_name])) + ";\n"
        + "\t\tmz= " + str(np.min(self.z_dict[object_name])) + " "
                + str(np.median(self.z_dict[object_name])) + " "
                + str(np.max(self.z_dict[object_name])) + ";\n"
        + "\t\tmφ= " + str(np.round(np.min(self.view_angle_dict[object_name]),2)) + " "
              + str(np.round(np.median(self.view_angle_dict[object_name]),2)) + " "
              + str(np.round(np.max(self.view_angle_dict[object_name]),2)) + ";\n"
        + "\t\tmY= " + str(np.min(self.yaw_dict[object_name])) + " "
                + str(np.median(self.yaw_dict[object_name])) + " "
                + str(np.max(self.yaw_dict[object_name])) + ";\n"
        + "\t\timage_height:= " + str(np.min(self.image_heights)) + " "
        + str(np.max(self.image_heights)) +";\n"
        + "\t\timage_width: " + str(np.min(self.image_widths)) + " "
        + str(np.max(self.image_widths)) + ";\n"
        "\t}\n")
      
    return statistics
  
  # ===========================================================================#
  # ================================== GETTERS ================================#
  # ===========================================================================#

  @property 
  def num_files(self):  return len(self._file_list)                             # get number of files in dataset


  def get_filename(self, frame_idx:int) -> str:
    """ Get the filename based on frame_idx.

    Parameters
    ----------
    frame_idx: int
      the index of the frame to get.

    Returns
    -------
    str
      a string containing the filename.
    """
    return self._file_list[frame_idx]
  

  def _get_file_list(self, image_dir:str) -> Tuple[str]:
    """Load all filenames from image_dir.

    Parameters
    __________
    image_dir: str
      path to the image directory

    Returns
    -------
    list 
      a list of all filenames in image directory
    """

    file_list = [f.split('.')[0] 
                 for f in os.listdir(image_dir) if isfile(join(image_dir, f))]
    file_list.sort()

    return file_list
  

  def _verify_file_list(self) -> Optional[ValueError]:
    """ Verify the files in file_list exist

    Raises
    ------
    assertion error when file in file_list is not complete.
    """

    def _(f):
      image_file = join(self._image_dir, f)+'.png'
      point_file = join(self._point_dir, f)+'.bin'
      label_file = join(self._label_dir, f)+'.txt'
      calib_file = join(self._calib_dir, f)+'.txt'

      assert isfile(image_file), "Image %s does not exist" % image_file
      assert isfile(point_file), "Point %s does not exist" % point_file

      if self._is_training:
        assert isfile(label_file), "Label %s does not exist" % label_file
      if not self._is_raw:
        assert isfile(calib_file), "Calib %s does not exist" % calib_file


    with ThreadPoolExecutor() as executor:
      executor.map(_, self._file_list)
        

  def get_calib(self, frame_idx:int) -> dict[np.array]:
    """Load calibration matrices and compute calibrations.

    Parameters
    ----------
    frame_idx: int
      the index of the frame to read.

    Returns
    -------
    dict[np.array]
      dictionary of calibration matrices.
    """

    calib_file = join(self._calib_dir, self._file_list[frame_idx])+'.txt'       # defuine calibaration files

    with open(calib_file, 'r') as f:                                            # laod calibration matrices         
      calib = {}
      for line in f:
        fields = line.split(' ')
        matrix_name = fields[0].rstrip(':')
        matrix = np.array(fields[1:], dtype=np.float32)
        calib[matrix_name] = matrix

    calib['P2']             = calib['P2'].reshape(3, 4)                         # get calibration matrices
    calib['R0_rect']        = calib['R0_rect'].reshape(3,3)
    calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3,4)
    R0_rect = np.eye(4)
    R0_rect[:3, :3]         = calib['R0_rect']
    calib['velo_to_rect']   = np.vstack([calib['Tr_velo_to_cam'],[0,0,0,1]])
    calib['cam_to_image']   = np.hstack([calib['P2'][:, 0:3], [[0],[0],[0]]])
    calib['rect_to_cam']    = np.hstack([calib['R0_rect'],
      np.matmul(np.linalg.inv(calib['P2'][:, 0:3]), calib['P2'][:, [3]])])
    calib['rect_to_cam']    = np.vstack([calib['rect_to_cam'],[0,0,0,1]])
    calib['velo_to_cam']    = np.matmul(
      calib['rect_to_cam'],calib['velo_to_rect'])
    calib['cam_to_velo']    = np.linalg.inv(calib['velo_to_cam'])

    calib['velo_to_image']  = np.matmul(                                        # sanity check
      calib['cam_to_image'],calib['velo_to_cam'])
    assert np.isclose(calib['velo_to_image'],
      np.matmul(np.matmul(calib['P2'], R0_rect),calib['velo_to_rect'])).all()
    
    return calib
  

  def get_velo_points(self, frame_idx:int, xyz_range:np.array=None) -> Points:
    """Load velo points from frame_idx

    Parameters
    ----------
    frame_idx: int
      the index of the frame to read.

    Returns
    -------
    Points
      velo points.
    """

    point_file  = join(self._point_dir, self._file_list[frame_idx])+'.bin'      # define point file
    velo_data   = np.fromfile(point_file, dtype=np.float32).reshape(-1, 4)      # load file as array
    velo_points = velo_data[:,:3]                                               # 3D coordinates
    reflections = velo_data[:,[3]]                                              # reflections

    mask = np.bool_(np.ones(len(velo_points)))                                  # filter points by range                     
    if xyz_range is not None:                                                          
      x_range, y_range, z_range = xyz_range
      mask *=(
          velo_points[:, 0] > x_range[0])*(velo_points[:, 0] < x_range[1])
      mask *=(
          velo_points[:, 1] > y_range[0])*(velo_points[:, 1] < y_range[1])
      mask *=(
          velo_points[:, 2] > z_range[0])*(velo_points[:, 2] < z_range[1])
      #return Points(xyz = velo_points[mask], attr = reflections[mask])
    
    return Points(xyz = velo_points[mask].astype(np.float64), attr = reflections[mask].astype(np.float64))
  

  def get_image(self, frame_idx:int) -> np.array:
    """ Load the image from frame_idx.

    Parameters
    ----------
    frame_idx: int 
      the index of the frame to read.
    
    Returns
    -------
    np.array
      image as numpy array
    """

    image_file = join(self._image_dir, self._file_list[frame_idx])+'.png'
    return cv2.imread(image_file)
  

  def get_label(self, frame_idx:int) -> Tuple[dict]:
    """Load bbox labels from frame_idx frame

    Parameters
    ----------
    frame_idx: int 
      the index of the frame to read.
    
    Returns
    -------
    list[dict]
      a list of object label dictionaries.
    """

    label_file = join(self._label_dir, self._file_list[frame_idx])+'.txt'       # define label file
    label_list = []                                                             # list of labels

    with open(label_file, 'r') as f:
      for line in f:                                                            # iterate label lines
        label = {}                                                              # label dictionary
        line  = line.strip()

        if line == '':  continue                                                # empty line
        
        fields = line.split(' ')                                                # define label properties
        label['name']       = str(fields[0])
        label['truncation'] = float(fields[1])
        label['occlusion']  = int(fields[2])
        label['alpha']      =  float(fields[3])
        label['xmin']       =  float(fields[4])
        label['ymin']       =  float(fields[5])
        label['xmax']       =  float(fields[6])
        label['ymax']       =  float(fields[7])
        label['height']     =  float(fields[8])
        label['width']      =  float(fields[9])
        label['length']     =  float(fields[10])
        label['x3d']        =  float(fields[11])
        label['y3d']        =  float(fields[12])
        label['z3d']        =  float(fields[13])
        label['yaw']        =  float(fields[14])

        if len(fields) > 15:  label['score'] =  float(fields[15])               # some labels seem to have scores

        if 0 <= self.difficulty <= 2:                                           # skip label not matching difficulty
          if label['truncation'] > TRUNCATION_THRESHOLDS[self.difficulty]:
            continue
          if label['occlusion'] > OCCULUSION_THRESHOLDS[self.difficulty]:
            continue
          if (label['ymax'] - label['ymin']) < \
            OBJECT_HEIGHT_THRESHOLDS[self.difficulty]: continue
        else:
          raise Exception("Unknown difficulty level: %d" % self.difficulty)
        label_list.append(label)

    return label_list
      

  def get_cam_points(self, frame_idx:int,downsample_voxel_size:float=None, 
                     calib:dict=None, xyz_range=None) -> Points:
    """Load velo points and convert them to (downsampled) camera coordinates
    
    Parameters
    ----------
    frame_idx: int
      the index of the frame to read.
    downsample_voxel_size:  float
      the size of voxel cells used for downsampling.
    calib: dict[np.ndarray]
      the calibration matrices.

    Returns
    -------
    Points
      camera points.
    """

    if calib is None: calib = self.get_calib(frame_idx)                         # get calibration matrices

    velo_points = self.get_velo_points(frame_idx, xyz_range=xyz_range)
    cam_points  = velo_to_cam(velo_points, calib)                               # convert velodyne to camera points

    if downsample_voxel_size is not None:
      cam_points = self.downsample_by_voxel(cam_points,downsample_voxel_size)   # downsample points by voxel
        
    return cam_points
  

  def sqdistance(self,p0:np.array,points:np.array) -> Tuple[float]:
    """ returns the squared distance between a point and a set of points 
    
    Paramters
    ---------
    p0: np.array
      a point
    
    points: np.array
      a set of points

    Returns
    -------
    float
      the squared distance between p0 and points.
    """

    return ((p0-points)**2).sum(axis=1)
  

  def get_cam_points_in_image(
    self, frame_idx:int, downsample_voxel_size:float=None,
    calib:dict=None, xyz_range:np.array=None
  ) -> Points:
    """ Load velo points and remove points that are not observed by camera.

    Parameters
    ----------
    frame_idx: int 
      index of the frame to read.
    downsample_voxel_size: float
      size of voxel cells used for downsampling.
    calib: dict[np.array]
      calibration matrices.
    xyz_range: np.ndarray
      range of xyz coordinates to filter.
    
    Returms
    -------
    image_points.
    """
    if calib is None: calib = self.get_calib(frame_idx)                         # load calibration matrices

    cam_points  = self.get_cam_points(                                          # get (downsampled) camera points 
      frame_idx, downsample_voxel_size,calib=calib, xyz_range=xyz_range
    )          
    image       = self.get_image(frame_idx)                                     # get frame image
    height      = image.shape[0]                                                # get image dimensions
    width       = image.shape[1]
    front_cam_points_idx  = cam_points.xyz[:,2] > 0.1                           
    filtered_points       = cam_points.xyz[front_cam_points_idx, :]             # get only points in front of camera
    filtered_attr         = None

    if cam_points.attr is not None:
      filtered_attr = cam_points.attr[front_cam_points_idx, :]

    front_cam_points  = Points(filtered_points,filtered_attr)
    img_points        = cam_points_to_image(front_cam_points, calib)            # transform to image points
    img_points_in_image_idx = np.logical_and.reduce(                            # filter points that are within image
      [img_points.xyz[:,0]>0, img_points.xyz[:,0]<width,
        img_points.xyz[:,1]>0, img_points.xyz[:,1]<height]
    )
    
    filtered_points   = front_cam_points.xyz[img_points_in_image_idx, :]
    filtered_attr     = None

    if front_cam_points.attr is not None:
      filtered_attr = front_cam_points.attr[img_points_in_image_idx, :]

    cam_points_in_img = Points(xyz=filtered_points, attr=filtered_attr)         # create new Points object of filtered objects 

    return cam_points_in_img
  

  def get_cam_points_in_image_with_rgb(
    self, frame_idx:int,downsample_voxel_size:float=None, calib:dict=None, 
    xyz_range:np.array=None
  ) -> Points:
    """Get camera points that are visible in image and append image color
    to the points as attributes
    
    Parameters
    ----------
    frame_idx: int
      index of the frame to read.
    downsample_voxel_size: float
      size of voxel cells used for downsampling.
    calib: dict[np.array]
      calibration matrices.
    xyz_range: np.array
      range of xyz coordinates to filter.

    Returns
    -------
    color encoded camera points.
    """

    if calib is None: calib = self.get_calib(frame_idx)                         # load calibration matrices

    cam_points_in_img = self.get_cam_points_in_image(
      frame_idx,downsample_voxel_size,calib,xyz_range)                          # get camera points in image
    image = self.get_image(frame_idx)     
    cam_points_in_img_with_rgb = self.rgb_to_cam_points(
      cam_points_in_img,image, calib)                                           # apply image colors to points
    
    return cam_points_in_img_with_rgb
  

  def boxes_3d_to_line_set(self, boxes_3d:list, boxes_color:list=None):
    """ takes in boxes_3d and returns corner points, lines and colors
    
    Parameters
    ----------
    boxes_3d:           Tuple[np.array] 
      list of 3D boxes
    boxes_color:        Tuple[np.array]
      list of colors for the boxes

    Returns
    -------
    Tuple[np.array]
      points, edges and colors of the boxes.
    """
    points  = []
    edges   = []
    colors  = []

    points  = boxes_3d_to_corners(boxes_3d)
    edges   = [
      np.array([
        [0, 1], [0, 4], [0, 3],
        [1, 2], [1, 5], [2, 3],
        [2, 6], [3, 7], [4, 5],
        [4, 7], [5, 6], [6, 7]
      ])+i*8 for i in range(len(boxes_3d))]
    
    if boxes_color is None:
      colors = [np.tile([[1.0, 0.0, 0.0]], [12, 1]) for i in range(len(boxes_3d))]
    else:
      colors = [np.tile(boxes_color[[i], :], [12, 1]) for i in range(len(boxes_3d))]

    if len(points) == 0:  return None, None, None                               # empty set

    return np.vstack(points), np.vstack(edges), np.vstack(colors)
    
    
  def get_open3D_box(
    self, label:dict, expend_factor:tuple=(1.0, 1.0, 1.0)
  ) -> Tuple[o3d.geometry.TriangleMesh]:
    """ creates o3d representation of bounding box

    Parameters
    ----------
    label: dict[float]
      dictionary containing "x3d", "y3d", "z3d", "yaw",
                  "height", "width", "lenth"
    expend_factor: Tuple[float]:
      tuple of (h, w, l) to expand the box.

    Returns
    -------
    Tuple[o3d.geometry.TriangleMesh]
      list of cylinders representing bounding box as a o3d mesh object.
    """
    yaw = label['yaw']                                                          # get label information
    h   = label['height']
    delta_h = h*(expend_factor[0]-1)
    w   = label['width']*expend_factor[1]
    l   = label['length']*expend_factor[2]
    tx  = label['x3d']
    ty  = label['y3d']
    tz  = label['z3d']
    Rh  = np.array([ 
      [1, 0, 0],
      [0, 0, 1],
      [0, 1, 0]])
    Rl = np.array([ 
      [0, 0, 1],
      [0, 1, 0],
      [1, 0, 0]])

     
    box_offset = BOX_OFFSET(l,w,h,delta_h)                                      # get box vertices  
    R = M_ROT(yaw)                                                              # define rotation matrix

    transform = np.matmul(R, np.transpose(box_offset))                          # rotate bounding box  
    transform = transform + np.array([[tx], [ty], [tz]])                        # translate from origin
    transform = np.vstack((transform, np.ones((1, 12))))                        # TODO: exmplanation missing !!
    hrotation = np.vstack((R.dot(Rh), np.zeros((1,3))))
    lrotation = np.vstack((R.dot(Rl), np.zeros((1,3))))
    wrotation = np.vstack((R, np.zeros((1,3))))
    box_color = [_/255 for _ in COLOR_MAP[label['name']][-1]]               # get box color
    box_color = [_/255 for _ in COLOR_MAP[label['name']][-1]]               # get box color

    h1_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=h/100, height=h)
    h1_cylinder.paint_uniform_color(box_color)
    h1_cylinder.transform(np.hstack((hrotation, transform[:, [0]])))

    h2_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=h/100, height=h)
    h2_cylinder.paint_uniform_color(box_color)
    h2_cylinder.transform(np.hstack((hrotation, transform[:, [1]])))

    h3_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=h/100, height=h)
    h3_cylinder.paint_uniform_color(box_color)
    h3_cylinder.transform(np.hstack((hrotation, transform[:, [2]])))

    h4_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=h/100, height=h)
    h4_cylinder.paint_uniform_color(box_color)
    h4_cylinder.transform(np.hstack((hrotation, transform[:, [3]])))

    w1_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=w/100, height=w)
    w1_cylinder.paint_uniform_color(box_color)
    w1_cylinder.transform(np.hstack((wrotation, transform[:, [4]])))

    w2_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=w/100, height=w)
    w2_cylinder.paint_uniform_color(box_color)
    w2_cylinder.transform(np.hstack((wrotation, transform[:, [5]])))

    w3_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=w/100, height=w)
    w3_cylinder.paint_uniform_color(box_color)
    w3_cylinder.transform(np.hstack((wrotation, transform[:, [6]])))

    w4_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=w/100, height=w)
    w4_cylinder.paint_uniform_color(box_color)
    w4_cylinder.transform(np.hstack((wrotation, transform[:, [7]])))

    l1_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=l/100, height=l)
    l1_cylinder.paint_uniform_color(box_color)
    l1_cylinder.transform(np.hstack((lrotation, transform[:, [8]])))

    l2_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=l/100, height=l)
    l2_cylinder.paint_uniform_color(box_color)
    l2_cylinder.transform(np.hstack((lrotation, transform[:, [9]])))

    l3_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=l/100, height=l)
    l3_cylinder.paint_uniform_color(box_color)
    l3_cylinder.transform(np.hstack((lrotation, transform[:, [10]])))

    l4_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=l/100, height=l)
    l4_cylinder.paint_uniform_color(box_color)
    l4_cylinder.transform(np.hstack((lrotation, transform[:, [11]])))

    return [
      h1_cylinder, h2_cylinder, h3_cylinder, h4_cylinder,
      w1_cylinder, w2_cylinder, w3_cylinder, w4_cylinder,
      l1_cylinder, l2_cylinder, l3_cylinder, l4_cylinder
    ]
  

  def sel_points_in_box3d(self, label:dict, points_xyz:np.array,
                          expend_factor:tuple=(1.0, 1.0, 1.0)) -> np.array:
    return sel_points_in_box3d(label, points_xyz, expend_factor)
  

  def sel_points_in_box2d(self, label:dict, points_xyz,
                          expend_factor:tuple=(1.0, 1.0)) -> np.array:
    return sel_points_in_box2d(label, points_xyz, expend_factor)
  

  def inspect_points(self, frame_idx, downsample_voxel_size=None, calib=None, 
                     expend_factor=(1.0, 1.0, 1.0), no_orientation=False):
    """ Visualize points in image 
    
    Paramters
    ---------
    frame_idx: int
      index of the frame to read.
    downsample_voxel_size: float
      size of voxel cells used for downsampling.
    calib: dict[np.array]
      calibration matrices.
    expend_factor: Tuple[float]
      tuple of (h, w, l) to expand the box.
    no_orientation: bool
      if True, draw 3D bounding boxes without orientation.
    """
    cam_points_in_img_with_rgb = self.get_cam_points_in_image_with_rgb(
      frame_idx, downsample_voxel_size=downsample_voxel_size, calib=calib)
    print("pc shape = "+str(cam_points_in_img_with_rgb.xyz.shape))
    label_list = self.get_label(frame_idx)
    self.vis_points(
      cam_points_in_img_with_rgb,label_list,expend_factor=expend_factor)
    

  def vis_points(
    self, cam_points_in_img_with_rgb,label_list  = None, 
    expend_factor=(1.0, 1.0, 1.0)):

    """ Visualize points in image with 3D bounding boxes

    Parameters
    ----------
    cam_points_in_img_with_rgb: Points
      Points object containing points and attributes.
    label_list: Tuple[dict]
      list of label dictionaries.
    expend_factor: Tuple[float]
      tuple of (h, w, l) to expand the box.
    """
    mesh_list   = []

    if label_list is not None:
      for label in label_list:                                                  # iterate all objects label list
        if label['name'] == 'DontCare': continue                                # skip DontCare objects
        point_mask = self.sel_points_in_box3d(
          label,cam_points_in_img_with_rgb.xyz,expend_factor=expend_factor)     # select points in bounding box 
        color = np.array(                                                       # get object color
          COLOR_MAP.get(label['name'], COLOR_MAP['KA'])[1])/255.0
        cam_points_in_img_with_rgb.attr[point_mask, 1:] = color                 # add colors to point attributes  
        mesh_list += self.get_open3D_box(label, expend_factor=expend_factor)    # add 3D bounding box to mesh list

    pcd = o3d.geometry.PointCloud()                                             # create point cloud object
    pcd.points = o3d.utility.Vector3dVector(cam_points_in_img_with_rgb.xyz)     # add points to point cloud
    pcd.colors = o3d.utility.Vector3dVector(                                    # add colors to point cloud
      cam_points_in_img_with_rgb.attr[:,1:4])
    
    def custom_draw_geometry_load_option(geometry_list):                        # visualize point cloud    
      vis = o3d.visualization.Visualizer()
      vis.create_window()
      for geometry in geometry_list:
        vis.add_geometry(geometry)
      ctr = vis.get_view_control()
      ctr.rotate(0.0, 3141.0, 0)
      vis.run()
      vis.destroy_window()

    custom_draw_geometry_load_option(mesh_list + [pcd])


  # ===========================================================================#
  # ================================== SETTERS ================================#
  # ===========================================================================#


  def downsample_by_voxel(self, points:Points, voxel_size:float, 
                          method:str='AVERAGE') -> Points:
    return downsample_by_voxel(points,voxel_size,method)
  

  def rgb_to_cam_points(
    self, points:Points, image:np.array, calib:dict) -> Points:
    """Append rgb colors to camera points attributes

    Parameters
    ----------
    points:   Points
      namedtuple containing "xyz" and "attr".
    image:    np.array
      numpy array containing image.
    calib:    dict[np.array] 
      dictionary containing calibration matrices.

    Returns
    -------
    Points namedtuple containing "xyz" and "attr" with color coding.
    """

    if calib is None: raise Exception('Calibration matrices are required')      # check calibration matrices

    img_points = cam_points_to_image(points, calib)
    rgb = image[np.int32(img_points.xyz[:,1]),
                np.int32(img_points.xyz[:,0]),::-1].astype(np.float32)/255
    
    if points.attr is None:                                                     # add rgb to points attributes
      return Points(points.xyz, rgb)                      
    else: 
      return Points(points.xyz, np.hstack([points.attr, rgb]))
  

  def vis_draw_2d_box(self, image:np.array, label_list:list):
    """ Draw 2D (min/max values) bounding boxes on the image with colors 
        indicating occlusion.

    Parameters
    ----------
    image:  np.array 
      image as np array.
    label_list:  Tuple[np.array]
      list of label dictionaries.
    """

    for label in label_list:
      if label['name'] == 'DontCare':                                           # set object color
        color = COLOR_MAP.get(label['name'])[1]      
      else: color = OCCLUSION_COLORS[label['occlusion']]

      xmin = int(label['xmin'])                                                 # get rectangle coordinates
      ymin = int(label['ymin'])
      xmax = int(label['xmax'])
      ymax = int(label['ymax'])

      cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)                # add bounding box to image
      cv2.putText(image, '{:s}'.format(label['name']),                          # add object name to image
                  (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,2)
      

  def velo_points_to_image(self, points:Points, calib:dict) -> np.array:
    """Convert points from velodyne coordinates to image coordinates. Points
    that behind the camera is removed.

    Parameters
    ----------
    points:  np.array
      [N, 3] float32 numpy array.
    calib:   dict[np.array]
      dictionary containing calibration matrices.

    Returns
    -------
      points on image plane: a [M, 2] float32 numpy array
    """

    cam_points = velo_to_cam(points, calib)
    img_points = cam_points_to_image(cam_points, calib)
    return img_points
      

  def vis_draw_3d_box(self, image, label_list, calib, color_map):
    """Draw 3D bounding boxes on the image.

    Parameters
    ----------
    image: np.array
      array containing the image.
    label_list:  Tuple[dict]
      list of label dictionaries.
    calib: dict[np.array]
      dictionary containing calibration matrices.
    color_map:  Tuple[np.array]
      dictionary containing color for each object.
    """
    for label in label_list:
      cam_points = box3d_to_cam_points(label)
      if any(cam_points.xyz[:, 2]<0.1): continue                                # only draw bb for objects in front of camera
      img_points    = cam_points_to_image(cam_points, calib)
      img_points_xy = img_points.xyz[:, 0:2].astype(np.int)
      color         = color_map[label['name']][::-1]

      cv2.line(image, tuple(img_points_xy[0,:]),                                # draw edges
               tuple(img_points_xy[1,:]),color,2)
      cv2.line(image, tuple(img_points_xy[1,:]),
               tuple(img_points_xy[5,:]),color,2)
      cv2.line(image, tuple(img_points_xy[5,:]),
               tuple(img_points_xy[4,:]),color,2)
      cv2.line(image, tuple(img_points_xy[4,:]),
               tuple(img_points_xy[0,:]),color,2)
      cv2.line(image, tuple(img_points_xy[1,:]),
               tuple(img_points_xy[2,:]),color,2)
      cv2.line(image, tuple(img_points_xy[2,:]),
               tuple(img_points_xy[6,:]),color,2)
      cv2.line(image, tuple(img_points_xy[6,:]),
               tuple(img_points_xy[5,:]),color,2)
      cv2.line(image, tuple(img_points_xy[2,:]),
               tuple(img_points_xy[3,:]),color,2)
      cv2.line(image, tuple(img_points_xy[3,:]),
               tuple(img_points_xy[7,:]),color,2)
      cv2.line(image, tuple(img_points_xy[7,:]),
               tuple(img_points_xy[6,:]),color,2)
      cv2.line(image, tuple(img_points_xy[3,:]),
               tuple(img_points_xy[0,:]),color,2)
      cv2.line(image, tuple(img_points_xy[4,:]),
               tuple(img_points_xy[7,:]),color,2)
      
  # ===========================================================================#
  # ============================ DATA AUGMENTATION ============================#
  # ===========================================================================#

  def save_cropped_boxes(
    self, expand_factor:tuple=(1.1, 1.1, 1.1),minimum_points:int=10,
    blacklist:list=['Van', 'Truck', 'Misc', 'Tram', 'Person_sitting','DonCare']
  ):
    """ creates a json file containing filtered labels and points within them

    Paramters
    ---------
    filename: str
      file to save the cropped boxes in
    expand_factor:  Tuple[float] 
      the expand factor for cropping
    minimum_points: int  
      minimum number of points in a cropped box
    backlist:  Tuple[str]
      list of str, the list of object names to be ignored
    """

    print('Cropping boxes and saving to %s' % self._crop_file)                  # print message

    def _(frame_idx):
      labels      = self.get_label(frame_idx)
      cam_points  = self.get_cam_points_in_image_with_rgb(frame_idx)
      ret_labels      = defaultdict(list) 
      ret_cam_points  = defaultdict(list)

      for label in labels:                                                      # iterate all labels                
        if label['name'] not in  blacklist:                                     # check if we care                                                         
          mask = self.sel_points_in_box3d(
            label, cam_points.xyz,expand_factor)                                # get mask of points in box

          if np.sum(mask) > minimum_points:                                     # sufficiently many points in bbox
            ret_labels[label['name']].append(label)                             # add label to dictionary
            ret_cam_points[label['name']].append(
              [cam_points.xyz[mask].tolist(),cam_points.attr[mask].tolist()]
            )
            return (ret_labels,ret_cam_points)
            
    cropped_labels      = defaultdict(list)
    cropped_cam_points  = defaultdict(list)

    with ThreadPoolExecutor(max_workers=10) as executor:
      futures = [executor.submit(_, item) for item in range(self.num_files)]
      results = []

      for future in tqdm(as_completed(futures), total=len(futures)):
        results.append(future.result())

      for result in results:
        if result is not None:
          for key in result[0]:
            cropped_labels[key].extend(result[0][key])
            cropped_cam_points[key].extend(result[1][key])
    # for frame_idx in tqdm(range(self.num_files)):
    #   cropped_labels[label['name']].append(label)

    #   cropped_cam_points[label['name']].append(
    #     [cam_points.xyz[mask].tolist(),cam_points.attr[mask].tolist()])
      

    with open(self._crop_file, 'w') as outfile:                                 # save cropped data to file
      json.dump((cropped_labels,cropped_cam_points), outfile)

  
  def load_cropped_boxes(self) -> Tuple[Tuple[dict],Points]:
    """ load cropped boxes from json file

    Paramters
    ---------
    filename: str 
      file to load

    Returns
    -------
      labels and points in the cropped boxes
    """
    try:                                                                        # load cropped data from file if exists else create it
      with open(self._crop_file, 'r') as infile:                                
        cropped_labels, cropped_cam_points = json.load(infile)
    except FileNotFoundError:
      self.save_cropped_boxes()
      with open(self._crop_file, 'r') as infile:                                
        cropped_labels, cropped_cam_points = json.load(infile)

    for key in cropped_cam_points:
      print("Loaded %d %s for augmentation" % (len(cropped_cam_points[key]), key))

      for i, cam_points in enumerate(cropped_cam_points[key]):                  # convert json to dictionary
        cropped_cam_points[key][i] = Points(xyz=np.array(cam_points[0]),
                                            attr=np.array(cam_points[1]))
          
    return cropped_labels, cropped_cam_points
  

  def vis_cropped_boxes(
    self,cropped_labels:dict, cropped_cam_points:dict, 
    object_class:str='Pedestrian'):
    """ Visualizes cropped boxes

    Paramters
    ---------
    cropped_labels: Tuple[dict]
      labels in the cropped boxes
    cropped_cam_points:  Points
      points in the cropped boxes
    object_class: str
      the object class to visualize
    """

    for key in cropped_cam_points:                                              # iteratte over unique object names
      if key == object_class:                                                   # visualize only selected
        for i, cam_points in enumerate(cropped_cam_points[key]):
          label = cropped_labels[key][i]
          print(label['name'])
          pcd = o3d.geometry.PointCloud()
          pcd.points = o3d.utility.Vector3dVector(cam_points.xyz)
          pcd.colors = o3d.utility.Vector3dVector(cam_points.attr[:, 1:])

          def custom_draw_geometry_load_option(geometry_list):
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            for geometry in geometry_list:  vis.add_geometry(geometry)
            ctr = vis.get_view_control()
            ctr.rotate(0.0, 3141.0, 0)
            vis.run()
            vis.destroy_window()

          custom_draw_geometry_load_option(                                     # draw points and boxes
            [pcd] + self.get_open3D_box(label))


  def parser_without_collision(
    self,cam_rgb_points:Points,labels:dict,sample_cam_points:Points,
    sample_labels:dict,overlap_mode:str='box',auto_box_height:bool=False,
    max_overlap_rate:float=0.01,appr_factor:int=100,max_overlap_num_allowed:int=1, 
    max_trails:int=1,method_name:str='normal',yaw_std:float=0.3, 
    expand_factor:tuple=(1.1, 1.1, 1.1),must_have_ground:bool=False
  ) -> Points:
    """ Parse cropped boxes to a frame without collision

    Paramters
    ---------
    cam_rgb_points:     Points	    
      Existing scene point cloud, a Points object (xyz coordinates + attributes).
    labels:  	          Tuple[np.array] 
      list of dictionaries describing existing objects 
      (3D boxes) already in the scene.
    sample_cam_points:	Points  
      Point clouds associated with new objects to be inserted.
    sample_labels: 	    Tuple[np.array] 
      Labels (box definitions) for the new objects to be inserted.
    overlap_mode:	      str
      How overlap is checked ("box", "point", or "both").
    auto_box_height:	  bool  
      Whether to auto-adjust box height to ground level.
    max_overlap_rate:	  float
      Maximum allowed overlap between boxes (for box mode).
    appr_factor:	      float
      Scaling factor applied to boxes when checking overlap.
    max_overlap_num_allowed:	int
      Maximum allowed number of overlapping points.
    max_trails:	        unsigned int
      Max retries per box to find a valid, non-overlapping position.
    method_name:	      str:
      How rotation is sampled (normal or uniform).
    yaw_std:	          float
      Standard deviation for random yaw sampling.
    expend_factor:	    Tuple[flaot]
      Expansion factor for boxes (for height adjustment and overlap checks).
    must_have_ground:	  bool
      If True, box is only valid if some ground points exist under it.

    Returns
    -------
    Points
      New scene point cloud with the new objects inserted.
    """
    xyz   = cam_rgb_points.xyz                                                  # extract points and attributes
    attr  = cam_rgb_points.attr

    if overlap_mode in ['box','box_and_point']:                                 # prepare bboxes for overlap checks
      scene_boxes = np.array(
        [[l['x3d'], l['y3d'], l['z3d'], l['length'],l['height'], 
          l['width'], l['yaw']] for l in labels ])
      scene_boxes_corners = np.int32(
        appr_factor*boxes_3d_to_corners(scene_boxes))
      
    for i, label in enumerate(sample_labels):                                   # iterate sample objects and try to place them in scene
      trial   = 0
      sucess  = False
      for trial in range(max_trails):
        if method_name == 'normal':                                             # define random rotation angle         
          delta_yaw = np.random.normal(scale=yaw_std)
        elif method_name == 'uniform':
          delta_yaw = np.random.uniform(low=-yaw_std, high=yaw_std)

        new_label = deepcopy(label)                                             # create new bbox label with random roation
        R   = M_ROT(delta_yaw)
        tx  = new_label['x3d']
        ty  = new_label['y3d']
        tz  = new_label['z3d']
        xyz_center = np.array([[tx, ty, tz]])
        xyz_center = xyz_center.dot(np.transpose(R))
        new_label['x3d'], new_label['y3d'], new_label['z3d'] = xyz_center[0]
        new_label['yaw'] = new_label['yaw']+delta_yaw

        if auto_box_height:                                                     # adjust height to ground
          mask_2d         = self.sel_points_in_box2d(
            new_label, xyz, expand_factor)

          if np.sum(mask_2d) > 0:
            ground_height = np.amax(xyz[mask_2d][:,1])
            y3d_adjust    = ground_height - new_label['y3d']
          else:
            if must_have_ground: continue
            y3d_adjust = 0
          # if np.abs(y3d_adjust) > 1:
          #     y3d_adjust = 0
          new_label['y3d'] += y3d_adjust

        mask = self.sel_points_in_box3d(new_label, xyz, expand_factor)
        below_overlap = False

        if not overlap_mode in ['box','point','box_and_point']:
          raise Exception(f'Unknown overlap mode: {overlap_mode}')
        
        below_overlap_b = True
        below_overlap_p = True

        if 'box' in overlap_mode:                                               # computes intersection rates with scene bboxes
          new_box = np.array([[
            new_label['x3d'],
            new_label['y3d'],
            new_label['z3d'],
            new_label['length'],
            new_label['height'],
            new_label['width'],
            new_label['yaw']
          ]])
          new_box_corners = np.int32(                                           # get bbox vertices
            appr_factor*boxes_3d_to_corners(new_box))
          below_overlap_b = np.all(overlapped_boxes_3d_fast_poly(               # compute overlap between new bbox and scene bboxes
            new_box_corners[0],
            scene_boxes_corners) < max_overlap_rate)
          
        if 'point' in overlap_mode:                                             # determines if max amount of scene points in box exceded
          below_overlap_p = np.sum(mask) < max_overlap_num_allowed

        below_overlap = below_overlap_b and below_overlap_p                     # get boolean condition according to overlap mode

        if below_overlap:
          points_xyz  = sample_cam_points[i].xyz
          points_attr = sample_cam_points[i].attr
          points_xyz  = points_xyz.dot(np.transpose(R))                         # rotate according to bbox rotation

          if auto_box_height: points_xyz[:,1] += y3d_adjust                     # align points to ground

          xyz   = xyz[np.logical_not(mask)]                                     # remove colliding scene points
          xyz   = np.concatenate([points_xyz, xyz], axis=0)
          attr  = attr[np.logical_not(mask)]
          attr  = np.concatenate([points_attr, attr], axis=0)
          
          labels.append(new_label)                                              # add new bbox to scene bboxes

          if overlap_mode in ['box','box_and_point']:                           # the added objects bbox matters in future insertion in this case
            if scene_boxes_corners.shape[0] > 0:
              scene_boxes_corners = np.append(scene_boxes_corners,
              new_box_corners,axis=0)
            else:
              scene_boxes_corners = np.append(np.empty((0,8,3)),                # the scene contains no prior bboxes
              new_box_corners,axis=0)
          sucess = True
          break
    # if not sucess:
        # if not sucess, keep the old label
        # print('Warning: fail to parse cropped box')
    return Points(xyz=xyz, attr=attr), labels


  def crop_aug(
    self, cam_rgb_points:Points, labels:dict,
    sample_rate:dict={"Car":1, "Pedestrian":1, "Cyclist":1},parser_kwargs:dict={}
  ) -> Points:
    """ uses cropped boxes to augment the scene

    Parameters
    ----------
    cam_rgb_points: Points
      point cloud in the frame
    labels:         Tuple[np.array] 
      list of labels in frame
    sample_rate:    dict 
      number of samples to take from each class
    parser_kwargs:  dict 
      arguments for the parser

    Returns
    -------
    Points
      points, cropped pointcloud
    """
    sample_labels     = []
    sample_cam_points = []

    for key in sample_rate:                                                     # get required number of object per class
      sample_indices = np.random.choice(len(self._cropped_labels[key]),
        size=sample_rate[key], replace=False)
      sample_labels.extend(
        deepcopy([self._cropped_labels[key][idx] for idx in sample_indices]))
      sample_cam_points.extend(
        deepcopy([self._cropped_cam_points[key][idx] for idx in sample_indices]))
      
    return self.parser_without_collision(
      cam_rgb_points    = cam_rgb_points,
      labels  = labels,
      sample_cam_points = sample_cam_points,
      sample_labels     = sample_labels,
      **parser_kwargs)


  def vis_crop_aug_sampler(self):
    """ Visualize the crop augmentation sampler """

    parser_kwargs = {
    'overlap_mode':     'box_and_point',
    'auto_box_height':  True,
    'max_overlap_rate': 1e-6,
    'appr_factor':      100,
    'max_overlap_num_allowed': 50,
    'max_trails':       100,
    'method_name':      'normal',
    'yaw_std':          np.pi/16,
    'expand_factor':    (1.1, 1.1, 1.1),
    'must_have_ground': True,
    }

    aug_config = {
      'method_name':    'random_box_global_rotation',
      'method_kwargs':  { 
        'max_overlap_num_allowed':100,
        'max_trails':     100,
        'method_name':    'normal',
        'yaw_std':        np.pi/8,
        'expend_factor':  (1.1, 1.1, 1.1)
      }
    }

    for frame_idx in range(10):
      labels          = self.get_label(frame_idx)
      cam_rgb_points  = self.get_cam_points_in_image_with_rgb(frame_idx)
      cam_rgb_points, labels = self.crop_aug(
        cam_rgb_points=cam_rgb_points, 
        labels=labels,
        sample_rate={"Car":2, "Pedestrian":10, "Cyclist":10},
        parser_kwargs=parser_kwargs)
      aug_configs = [aug_config]
      aug_fn      = get_data_aug(aug_configs)
      cam_rgb_points, labels = aug_fn(cam_rgb_points, labels)
      labels = list(filter(lambda l: l['name'] != 'DontCare', labels))
      self.vis_points(cam_rgb_points, labels, expend_factor=(1.1, 1.1,1.1))


  def visualize_graph(self,frame_idx,show_bboxes=True):
    """
    Visualizes the graph using Open3D.

    Parameters
    ----------
    base_points : np.ndarray
      The base points of the graph.
    ds_points : np.ndarray
      The downsampled points of the graph.
    edge_list : list
      The list of edges in the graph.

    Returns
    -------
    None
    """

    with open('configs/car_auto_T0_config','r') as f:  
      config = json.load(f)
      
    graph_gen_configs = config['graph_gen_kwargs']
    level_configs     = graph_gen_configs['level_configs']

    base_voxel_size   = graph_gen_configs['base_voxel_size']
    downsample_method = graph_gen_configs['downsample_method']

    points = self.get_cam_points_in_image_with_rgb(frame_idx,0.01)
    #points = points[np.where(points[:,2] > 0)]

    _,edge_list = gen_multi_level_local_graph_v3(                               # compute graph vertices and edges
      points_xyz        = points.xyz,
      base_voxel_size   = base_voxel_size,
      level_configs     = level_configs,
      downsample_method = downsample_method
    )

    mesh_list = []
    if show_bboxes:                                                             # conditionally add 3D bounding boxes.
      labels    = self.get_label(frame_idx)
      for i,label in enumerate(labels):
        mesh_list = mesh_list + self.get_open3D_box(label)

    xyz     = points.xyz
    colors  = points.attr[:,1:4]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    edges   = edge_list[0]
    colors  = (colors[edges[:,0]]+ colors[edges[:,1]])/2

    line_set    = o3d.geometry.LineSet()
    line_set.points  = o3d.utility.Vector3dVector(points.xyz)
    line_set.lines   = o3d.utility.Vector2iVector(edge_list[0])
    line_set.colors  = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geometry in (mesh_list+[pcd]+[line_set]):
      vis.add_geometry(geometry)
    ctr = vis.get_view_control()
    ctr.rotate(0.0,3141.0,0)
    vis.run()
    vis.destroy_window()

  # ===========================================================================#
  # ========================== POINT CLASS ASSIGNMENT =========================#
  # ===========================================================================#

  def assign_classaware_label_to_points(self, labels, xyz, expend_factor,label_method=['Car']):
    """ Assign class-aware labels to points in the point cloud.
    
    Parameters
    ----------
    labels: list[dict]
      list of dictionaries containing object labels.
    xyz: np.array
      point cloud coordinates.
    expend_factor: tuple[float]
      tuple of (h, w, l) to expand the box.
    
    Returns
    -------
    cls_labels: np.array
      class labels for each point.
    boxes_3d: np.array
      3D bounding boxes in scene.
    valid_boxes: np.array
      mask for boxes we care about.
    """
    num_points = xyz.shape[0]
    assert num_points > 0, "No point No prediction"
    assert xyz.shape[1] == 3, 'Invalid point shapes'

    cls_labels  = LABEL_MAP['Background']*np.ones(                              # default label is Background
      num_points, dtype=np.int16)                      
    
    boxes_3d    = np.zeros((num_points, 7))                                     # 3d boxes for each point 
    valid_boxes = np.zeros(num_points, dtype=np.int16)                          # mask for boxes we care about  
    
    for label in labels:                                                        # add label to each object
      obj_cls_string = label['name']
      obj_cls = LABEL_MAP.get(obj_cls_string, LABEL_MAP['DontCare'])
      mask    = self.sel_points_in_box3d(label, xyz, expend_factor)

      if obj_cls in [LABEL_MAP[obj] for obj in label_method]:                  # check if object is a car
        yaw = label['yaw']

        while yaw < -0.25*np.pi:    yaw += np.pi                                # restrict yaw
        while yaw > 0.75*np.pi:     yaw -= np.pi

        if yaw < 0.25*np.pi: cls_labels[mask] = obj_cls                         # horizontal
        else: cls_labels[mask] = obj_cls+1                                      # vertical

        boxes_3d[mask,:] = (                                                    # append label
          label['x3d'], label['y3d'],label['z3d'],label['length'], 
          label['height'],label['width'], yaw
        )
        valid_boxes[mask] = 1                                                   # update mask
      else:
        if obj_cls_string != 'DontCare':
          cls_labels[mask] = obj_cls
          valid_boxes[mask] = 0
                
    return cls_labels, boxes_3d, valid_boxes
  