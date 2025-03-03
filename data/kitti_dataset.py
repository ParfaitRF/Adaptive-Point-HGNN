import os
from os.path import isfile, join
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d

from globals import (
  M_ROT,BOX_OFFSET,
  IMG_WIDTH, IMG_HEIGHT,                                                        # image dimensions
  OBJECT_HEIGHT_THRESHOLDS,TRUNCATION_THRESHOLDS,OCCULUSION_THRESHOLDS,         # thresholds
  OCCLUSION_COLORS,OBJECT_COLORS                                                # colors
)
from .transformations import (
  boxes_3d_to_corners,box3d_to_cam_points,box3d_to_normals,
  cam_points_to_image,velo_to_cam,
)
from .utils import (
  Points,
  downsample_by_average_voxel,sel_points_in_box3d,
  sel_points_in_box2d,downsample_by_voxel
)


class KittiDataset(object):
  """ A class for interactions with the KITTI dataset """
# =============================================================================#
# ============================== INITIALIZATION ===============================#
# =============================================================================#

  def __init__(
    self, index_filename:str=None, is_training:bool=True, is_raw:bool=False, 
    difficulty:int=0,num_classes:int=8):
    """ Constructor of KittiDataset.

    @param image_dir:   path to image folder.ooo
    @param point_dir:   path to point cloud data folder.
    @param calib_dir:   path to the calibration matrices.
    @param label_dir:   path to the label folder.
    @param index_filename:  path an index file.
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
    data_root_dir = join(data_root_dir,'kitti')
    subfolder     = 'training' if is_training else 'testing'

    self._calib_dir = join(data_root_dir,subfolder,'calib')
    self._image_dir = join(data_root_dir,subfolder,'image_2')
    self._point_dir = join(data_root_dir,subfolder,'velodyne')
    if is_training: 
      self._label_dir = join(data_root_dir,subfolder,'label_2')

    self._file_list = self._get_file_list(self._image_dir)

    self._verify_file_list()
    
    # set the maximum image height and width
    self._max_image_height  = IMG_HEIGHT
    self._max_image_width   = IMG_WIDTH

  # ===========================================================================#
  # =========================== STRING REPRESENTATION =========================#
  # ===========================================================================#

  def __str__(self):
    """ Generate a string summary of the dataset """

    summary_string = ('Dataset Summary:\n'
      + '* Paths{\n'
      + '\timage_dir=%s\n' % self._image_dir
      + '\tpoint_dir=%s\n' % self._point_dir
      + '\tcalib_dir=%s\n' % self._calib_dir
      + '\tlabel_dir=%s\n' % self._label_dir
      + '}\n\n'
      + '* Total number of sampels: %d {\n' % self.num_files)
    statics = self.get_statics()

    return summary_string + statics
  

  def get_statics(self):
    """ Get statistics of objects in the dataset """

    # coordinates lists
    x_dict = defaultdict(list)
    y_dict = defaultdict(list)
    z_dict = defaultdict(list)
    # bounding box dimensions list
    h_dict = defaultdict(list)
    w_dict = defaultdict(list)
    l_dict = defaultdict(list)
    view_angle_dict = defaultdict(list) # view angle list
    yaw_dict        = defaultdict(list) # yaw list

    for frame_idx in range(self.num_files):
      labels = self.get_label(frame_idx)                                        # get labels for in frame
      for label in labels:                                                      # get all label information in frame                    
        if label['ymin'] > 0:
          if label['ymax'] - label['ymin'] > \
            OBJECT_HEIGHT_THRESHOLDS[self.difficulty]:                          # large enough
            object_name = label['name']
            h_dict[object_name].append(label['height'])                         # object height
            w_dict[object_name].append(label['width'])                          # object width
            l_dict[object_name].append(label['length'])                         # object length
            x_dict[object_name].append(label['x3d'])                            # x coordinate
            y_dict[object_name].append(label['y3d'])                            # y coordinate
            z_dict[object_name].append(label['z3d'])                            # z coordinate
            view_angle_dict[object_name].append(                                # compute view angle  
              np.arctan(label['x3d']/label['z3d']))
            yaw_dict[object_name].append(label['yaw'])

    plt.scatter(z_dict['Pedestrian'], np.array(l_dict['Pedestrian']))           # plot scatter plot of pedestrians
    plt.title('Scatter plot pythonspot.com')
    plt.show()

    # compute ingore statics
    truncation_rates    = []
    no_truncation_rates = []
    image_height        = []
    image_width         = []

    for frame_idx in range(self.num_files):
      labels  = self.get_label(frame_idx)                                       # get labels for in frame
      calib   = self.get_calib(frame_idx)
      image   = self.get_image(frame_idx)
      image_height.append(image.shape[0])
      image_width.append(image.shape[1])

      for label in labels:
        object_name = label['name']

        if label['name'] == 'Car':
          if label['ymax'] - label['ymin'] < OBJECT_HEIGHT_THRESHOLDS[-1]:      # too small
            h_dict['ignored_by_height'].append(label['height'])
            w_dict['ignored_by_height'].append(label['width'])
            l_dict['ignored_by_height'].append(label['length'])
            x_dict['ignored_by_height'].append(label['x3d'])
            y_dict['ignored_by_height'].append(label['y3d'])
            z_dict['ignored_by_height'].append(label['z3d'])
            view_angle_dict['ignored_by_height'].append(
               np.arctan(label['x3d']/label['z3d']))
            yaw_dict['ignored_by_height'].append(label['yaw'])

          if label['truncation'] > TRUNCATION_THRESHOLDS[self.difficulty]:      # too much truncation
            h_dict['ignored_by_truncation'].append(label['height'])
            w_dict['ignored_by_truncation'].append(label['width'])
            l_dict['ignored_by_truncation'].append(label['length'])
            x_dict['ignored_by_truncation'].append(label['x3d'])
            y_dict['ignored_by_truncation'].append(label['y3d'])
            z_dict['ignored_by_truncation'].append(label['z3d'])
            view_angle_dict['ignored_by_truncation'].append(
              np.arctan(label['x3d']/label['z3d']))
            yaw_dict['ignored_by_truncation'].append(label['yaw'])

          detection_boxes_3d = np.array(                                        # get label 3D boxes
            [[label['x3d'], label['y3d'], label['z3d'], label['length'], 
              label['height'], label['width'], label['yaw']]])
          detection_boxes_3d_corners = boxes_3d_to_corners(detection_boxes_3d)  # translate from origin
          corners_cam_points = Points(                                          # convert to Points object  
            xyz=detection_boxes_3d_corners[0], attr=None)
          corners_img_points = cam_points_to_image(corners_cam_points, calib)   # convert velodyne into image points
          corners_xy  = corners_img_points.xyz[:, :2]                           # get x and y coordinates         
          xmin, ymin  = np.amin(corners_xy, axis=0)                             # get min and max coordinates
          xmax, ymax  = np.amax(corners_xy, axis=0)
          clip_xmin   = max(xmin, 0.0)                                          # define clip off points
          clip_ymin   = max(ymin, 0.0)
          clip_xmax   = min(xmax, IMG_WIDTH)
          clip_ymax   = min(ymax, IMG_HEIGHT)
          truncation_rate = 1.0 - \
            (clip_ymax - clip_ymin)*(clip_xmax - clip_xmin)/\
              ((ymax - ymin)*(xmax - xmin))                                     # compute truncation rate
          
          if label['truncation'] > TRUNCATION_THRESHOLDS[self.difficulty]:      # check truncation rate
            truncation_rates.append(truncation_rate)
          else:
            no_truncation_rates.append(truncation_rate)

          if label['occlusion'] > OCCULUSION_THRESHOLDS[self.difficulty]:       # check occlusion rate
            h_dict['ignored_by_occlusion'].append(label['height'])
            w_dict['ignored_by_occlusion'].append(label['width'])
            l_dict['ignored_by_occlusion'].append(label['length'])
            x_dict['ignored_by_occlusion'].append(label['x3d'])
            y_dict['ignored_by_occlusion'].append(label['y3d'])
            z_dict['ignored_by_occlusion'].append(label['z3d'])
            view_angle_dict['ignored_by_occlusion'].append(
              np.arctan(label['x3d']/label['z3d']))
            yaw_dict['ignored_by_occlusion'].append(label['yaw'])

    statistics = ""
    for object_name in h_dict:
      # print(object_name+
      #   "l="+str(np.histogram(l_dict[object_name], 10, density=True))+'\n')
      
      if len(h_dict[object_name]) == 0: continue

      statistics += ('\t* ' + str(object_name) + 's {\n' 
        + '\t\t# objects= ' + str(len(h_dict[object_name])) + ";\n"
        + "\t\tmh= " + str(np.min(h_dict[object_name])) + " "
                + str(np.median(h_dict[object_name])) + " "
                + str(np.max(h_dict[object_name])) + ";\n"
        + "\t\tmw= " + str(np.min(w_dict[object_name])) + " "
                + str(np.median(w_dict[object_name])) + " "
                + str(np.max(w_dict[object_name])) + ";\n"
        + "\t\tml= " + str(np.min(l_dict[object_name])) + " "
                + str(np.median(l_dict[object_name])) + " "
                + str(np.max(l_dict[object_name])) + ";\n"
        + "\t\tmx= " + str(np.min(x_dict[object_name])) + " "
                + str(np.median(x_dict[object_name])) + " "
                + str(np.max(x_dict[object_name])) + ";\n"
        + "\t\tmy= " + str(np.min(y_dict[object_name])) + " "
                + str(np.median(y_dict[object_name])) + " "
                + str(np.max(y_dict[object_name])) + ";\n"
        + "\t\tmz= " + str(np.min(z_dict[object_name])) + " "
                + str(np.median(z_dict[object_name])) + " "
                + str(np.max(z_dict[object_name])) + ";\n"
        + "\t\tmA= " + str(np.round(np.min(view_angle_dict[object_name]),2)) + " "
              + str(np.round(np.median(view_angle_dict[object_name]),2)) + " "
              + str(np.round(np.max(view_angle_dict[object_name]),2)) + ";\n"
        + "\t\tmY= " + str(np.min(yaw_dict[object_name])) + " "
                + str(np.median(yaw_dict[object_name])) + " "
                + str(np.max(yaw_dict[object_name])) + ";\n"
        + "\t\timage_height:= " + str(np.min(image_height)) + " "
        + str(np.max(image_height)) +";\n"
        + "\t\timage_width: " + str(np.min(image_width)) + " "
        + str(np.max(image_width)) + ";\n"
        "\t}\n")
      
    return statistics
  
  # ===========================================================================#
  # ================================== GETTERS ================================#
  # ===========================================================================#

  @property 
  def num_files(self):  return len(self._file_list)                             # get number of files in dataset


  def get_filename(self, frame_idx):
    """ Get the filename based on frame_idx.

    @param frame_idx: the index of the frame to get.
    @return: a string containing the filename.
    """
    return self._file_list[frame_idx]
  

  def _get_file_list(self, image_dir:str):
    """Load all filenames from image_dir.

    @param image_dir: path to the image directory

    Returns: a list of all filenames in image directory
    """

    file_list = [f.split('.')[0] 
                 for f in os.listdir(image_dir) if isfile(join(image_dir, f))]
    file_list.sort()

    return file_list
  

  def _verify_file_list(self):
    """ Verify the files in file_list exist

    @raise: assertion error when file in file_list is not complete.
    """

    for f in self._file_list:
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
      

  def get_calib(self, frame_idx:int):
    """Load calibration matrices and compute calibrations.

    @param frame_idx: the index of the frame to read.
    @return: a dictionary of calibrations.
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
  

  def get_velo_points(self, frame_idx:int, xyz_range:np.array=None):
    """Load velo points from frame_idx

    @param frame_idx: the index of the frame to read.
    @return: Points object containing the velo points.
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
    
    return Points(xyz = velo_points[mask], attr = reflections[mask])
  

  def get_image(self, frame_idx:int):
    """Load the image from frame_idx.

    @param frame_idx: the index of the frame to read.
    @return: cv2.matrix
    """

    image_file = join(self._image_dir, self._file_list[frame_idx])+'.png'
    return cv2.imread(image_file)
  

  def get_label(self, frame_idx:int):
    """Load bbox labels from frame_idx frame

    @param frame_idx: the index of the frame to read.
    @return: a list of object label dictionaries.
    """

    label_file = join(self._label_dir, self._file_list[frame_idx])+'.txt'       # define label file
    label_list = []                                                             # list of labels

    with open(label_file, 'r') as f:
      for line in f:                                                            # iterate label lines
        label={}                                                                # label dictionary
        line = line.strip()

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
                     calib:dict=None, xyz_range=None):
    """Load velo points and convert them to (downsampled) camera coordinates

    @param frame_idx: the index of the frame to read.
    @param downsample_voxel_size: the size of voxel cells used for downsampling.
    @param calib: the calibration matrices.
    @return: Points object containing the camera points.
    """

    if calib is None: calib = self.get_calib(frame_idx)                         # get calibration matrices

    velo_points = self.get_velo_points(frame_idx, xyz_range=xyz_range)
    cam_points  = velo_to_cam(velo_points, calib)                           # convert velodyne to camera points

    if downsample_voxel_size is not None:
      cam_points = self.downsample_by_voxel(cam_points,downsample_voxel_size) # downsample points by voxel
        
    return cam_points
  

  def sqdistance(self,p0:np.array,points:np.array):
    """ returns the squared distance between a point and a set of points 
    @param p0: a point
    @param points: a set of points
    """

    return ((p0-points)**2).sum(axis=1)
  

  def get_cam_points_in_image(self, frame_idx:int, downsample_voxel_size:float=None,
    calib:dict=None, xyz_range:np.array=None):
    """Load velo points and remove points that are not observed by camera.
    """
    if calib is None: calib = self.get_calib(frame_idx)                         # load calibration matrices

    cam_points = self.get_cam_points(frame_idx, downsample_voxel_size,
                                     calib=calib, xyz_range=xyz_range)          # get (downsampled) camera points 
    image   = self.get_image(frame_idx)                                         # get frame image
    height  = image.shape[0]                                                    # get image dimensions
    width   = image.shape[1]
    front_cam_points_idx  = cam_points.xyz[:,2] > 0.1                           # I think this filters points that are within some depth ??
    filtered_points = cam_points.xyz[front_cam_points_idx, :]
    filtered_attr   = None
    if cam_points.attr is not None:
      filtered_attr = cam_points.attr[front_cam_points_idx, :]
    front_cam_points  = Points(filtered_points,filtered_attr)
    img_points        = cam_points_to_image(front_cam_points, calib)              # transform to image points
    img_points_in_image_idx = np.logical_and.reduce(                            # filter points that are within image
      [img_points.xyz[:,0]>0, img_points.xyz[:,0]<width,
        img_points.xyz[:,1]>0, img_points.xyz[:,1]<height])
    
    filtered_points   = front_cam_points.xyz[img_points_in_image_idx, :]
    filtered_attr     = None
    if front_cam_points.attr is not None:
      filtered_attr = front_cam_points.attr[img_points_in_image_idx, :]
    cam_points_in_img = Points(xyz=filtered_points, attr=filtered_attr)         # create new Points object consisting of filtered objects 

    return cam_points_in_img
  

  def get_cam_points_in_image_with_rgb(self, frame_idx:int,
    downsample_voxel_size:float=None, calib:dict=None, xyz_range:np.array=None):
    """Get camera points that are visible in image and append image color
    to the points as attributes
    
    @param frame_idx: the index of the frame to read.
    @param downsample_voxel_size: the size of voxel cells used for downsampling.
    @param calib: the calibration matrices.
    @param xyz_range: the range of xyz coordinates to filter.
    @return: Points object containing the camera points.
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
    
    @param boxes_3d: a list of 3D boxes
    @param boxes_color: a list of colors for the boxes
    """
    points  = []
    edges   = []
    colors  = []

    for i, box_3d in enumerate(boxes_3d):
      x3d, y3d, z3d, l, h, w, yaw = box_3d                                      # get box information
      
      corners = np.array([                                                      # get corners of 3D box
        [ l/2,  0.0,  w/2], # front up right
        [ l/2,  0.0, -w/2], # front up left
        [-l/2,  0.0, -w/2], # back up left
        [-l/2,  0.0,  w/2], # back up right
        [ l/2, -h,  w/2],   # front down right
        [ l/2, -h, -w/2],   # front down left
        [-l/2, -h, -w/2],   # back down left
        [-l/2, -h,  w/2]]   # back down right
      ) 
      R = M_ROT(yaw)
      r_corners       = corners.dot(np.transpose(R))                            # rotate corners              
      cam_points_xyz  = r_corners+np.array([x3d, y3d, z3d])                     # translate from origin
      points.append(cam_points_xyz)                                             
      edges.append(                                                             # define edges
        np.array([
          [0, 1], [0, 4], [0, 3],
          [1, 2], [1, 5], [2, 3],
          [2, 6], [3, 7], [4, 5],
          [4, 7], [5, 6], [6, 7]
        ])+i*8
      )
      
      if boxes_color is None:                                                   # define edges color
        colors.append(np.tile([[1.0, 0.0, 0.0]], [12, 1]))
      else:
        colors.append(np.tile(boxes_color[[i], :], [12, 1]))

    if len(points) == 0:  return None, None, None                               # empty set

    return np.vstack(points), np.vstack(edges), np.vstack(colors)
    

  def get_open3D_box(self, label:dict, expend_factor:tuple=(1.0, 1.0, 1.0)):
    """ creates o3d representation of bounding box

    @param label: a dictionary containing "x3d", "y3d", "z3d", "yaw",
                  "height", "width", "lenth"
    @param expend_factor: a tuple of (h, w, l) to expand the box.
    @return: a o3d mesh object.
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
    box_color = [_/255 for _ in OBJECT_COLORS[label['name']][-1]]               # get box color

    h1_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=h/100, height=h)
    #h1_cylinder = o3d.create_mesh_cylinder(radius = h/100, height = h)          # create edges
    h1_cylinder.paint_uniform_color(box_color)
    h1_cylinder.transform(np.hstack((hrotation, transform[:, [0]])))

    #h2_cylinder = o3d.create_mesh_cylinder(radius = h/100, height = h)
    h2_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=h/100, height=h)
    h2_cylinder.paint_uniform_color(box_color)
    h2_cylinder.transform(np.hstack((hrotation, transform[:, [1]])))

    #h3_cylinder = o3d.create_mesh_cylinder(radius = h/100, height = h)
    h3_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=h/100, height=h)
    h3_cylinder.paint_uniform_color(box_color)
    h3_cylinder.transform(np.hstack((hrotation, transform[:, [2]])))

    #h4_cylinder = o3d.create_mesh_cylinder(radius = h/100, height = h)
    h4_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=h/100, height=h)
    h4_cylinder.paint_uniform_color(box_color)
    h4_cylinder.transform(np.hstack((hrotation, transform[:, [3]])))

    #w1_cylinder = o3d.create_mesh_cylinder(radius = w/100, height = w)
    w1_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=w/100, height=w)
    w1_cylinder.paint_uniform_color(box_color)
    w1_cylinder.transform(np.hstack((wrotation, transform[:, [4]])))

    #w2_cylinder = o3d.create_mesh_cylinder(radius = w/100, height = w)
    w2_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=w/100, height=w)
    w2_cylinder.paint_uniform_color(box_color)
    w2_cylinder.transform(np.hstack((wrotation, transform[:, [5]])))

    #w3_cylinder = o3d.create_mesh_cylinder(radius = w/100, height = w)
    w3_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=w/100, height=w)
    w3_cylinder.paint_uniform_color(box_color)
    w3_cylinder.transform(np.hstack((wrotation, transform[:, [6]])))

    #w4_cylinder = o3d.create_mesh_cylinder(radius = w/100, height = w)
    w4_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=w/100, height=w)
    w4_cylinder.paint_uniform_color(box_color)
    w4_cylinder.transform(np.hstack((wrotation, transform[:, [7]])))

    #l1_cylinder = o3d.create_mesh_cylinder(radius = l/100, height = l)
    l1_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=l/100, height=l)
    l1_cylinder.paint_uniform_color(box_color)
    l1_cylinder.transform(np.hstack((lrotation, transform[:, [8]])))

    #l2_cylinder = o3d.create_mesh_cylinder(radius = l/100, height = l)
    l2_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=l/100, height=l)
    l2_cylinder.paint_uniform_color(box_color)
    l2_cylinder.transform(np.hstack((lrotation, transform[:, [9]])))

    #l3_cylinder = o3d.create_mesh_cylinder(radius = l/100, height = l)
    l3_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=l/100, height=l)
    l3_cylinder.paint_uniform_color(box_color)
    l3_cylinder.transform(np.hstack((lrotation, transform[:, [10]])))

    #l4_cylinder = o3d.create_mesh_cylinder(radius = l/100, height = l)
    l4_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=l/100, height=l)
    l4_cylinder.paint_uniform_color(box_color)
    l4_cylinder.transform(np.hstack((lrotation, transform[:, [11]])))

    return [
      h1_cylinder, h2_cylinder, h3_cylinder, h4_cylinder,
      w1_cylinder, w2_cylinder, w3_cylinder, w4_cylinder,
      l1_cylinder, l2_cylinder, l3_cylinder, l4_cylinder
    ]
  

  def sel_points_in_box3d(self, label:dict, points:Points,
                          expend_factor:tuple=(1.0, 1.0, 1.0)):
    return sel_points_in_box3d(label, points, expend_factor)
  

  def sel_points_in_box2d(self, label:dict, points:Points,
                          expend_factor:tuple=(1.0, 1.0)):
    return sel_points_in_box2d(label, points, expend_factor)
  

  def inspect_points(self, frame_idx, downsample_voxel_size=None, calib=None, 
                     expend_factor=(1.0, 1.0, 1.0), no_orientation=False):
    """ Visualize points in image 
    
    @param frame_idx: the index of the frame to read.
    @param downsample_voxel_size: the size of voxel cells used for downsampling.
    @param calib: the calibration matrices.
    @param expend_factor: a tuple of (h, w, l) to expand the box.
    @param no_orientation: if True, draw 3D bounding boxes without orientation.
    """
    cam_points_in_img_with_rgb = self.get_cam_points_in_image_with_rgb(
      frame_idx, downsample_voxel_size=downsample_voxel_size, calib=calib)
    print("#(points)="+str(cam_points_in_img_with_rgb.xyz.shape))
    label_list = self.get_label(frame_idx)
    self.vis_points(cam_points_in_img_with_rgb,label_list, 
                    expend_factor=expend_factor)
    

  def vis_points(self, cam_points_in_img_with_rgb,
    label_list  = None, expend_factor=(1.0, 1.0, 1.0)):
    mesh_list   = []

    if label_list is not None:
      for label in label_list:                                                  # iterate all objects label list
        #print(label['name'])
        point_mask = self.sel_points_in_box3d(
          label,cam_points_in_img_with_rgb.xyz,expend_factor=expend_factor)     # select points in bounding box 
        color = np.array(                                                       # get object color
          OBJECT_COLORS.get(label['name'], ["Olive",(0,128,0)])[1])/255.0
        cam_points_in_img_with_rgb.attr[point_mask, 1:] = color                 # add colors to point attributes  
        mesh_list += self.get_open3D_box(label, expend_factor=expend_factor)    # add 3D bounding box to mesh list

    pcd = o3d.geometry.PointCloud()                                                      # create point cloud object
    pcd.points = o3d.utility.Vector3dVector(cam_points_in_img_with_rgb.xyz)             # add points to point cloud
    pcd.colors = o3d.utility.Vector3dVector(                                            # add colors to point cloud
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
                          method:str='AVERAGE'):
    return downsample_by_voxel(points,voxel_size,method='AVERAGE')
  

  def rgb_to_cam_points(self, points:Points, image:np.array, calib:dict):
    """Append rgb colors to camera points attributes
    
    @param points:  a Points namedtuple containing "xyz" and "attr".
    @param image:   a numpy array containing the image.
    @param calib:   a dictionary containing calibration matrices.
    @return points: a Points namedtuple containing "xyz" and "attr".
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

    @param image:       a numpy array containing the image.
    @param label_list:  a list of label dictionaries.
    """

    for label in label_list:
      if label['name'] == 'DontCare':                                           # set object color
        color = OBJECT_COLORS.get(label['name'])[1]      
      else: color = OCCLUSION_COLORS[label['occlusion']]

      xmin = int(label['xmin'])                                                 # get rectangle coordinates
      ymin = int(label['ymin'])
      xmax = int(label['xmax'])
      ymax = int(label['ymax'])

      cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)                # add bounding box to image
      cv2.putText(image, '{:s}'.format(label['name']),                          # add object name to image
                  (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,2)
      

  def velo_points_to_image(self, points:Points, calib:dict):
    """Convert points from velodyne coordinates to image coordinates. Points
    that behind the camera is removed.

    @param points:  a [N, 3] float32 numpy array.
    @param calib:   a dictionary containing calibration matrices.
    @return points on image plane: a [M, 2] float32 numpy array,
            a mask indicating points: a [N, 1] boolean numpy array.
    """

    cam_points = velo_to_cam(points, calib)
    img_points = cam_points_to_image(cam_points, calib)
    return img_points
      

  def vis_draw_3d_box(self, image, label_list, calib, color_map):
    """Draw 3D bounding boxes on the image.

    @param image:       a numpy array containing the image.
    @param label_list:  a list of label dictionaries.
    @param calib:       a dictionary containing calibration matrices.
    @param color_map:   a dictionary containing color for each object.
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
      

  
  
  

