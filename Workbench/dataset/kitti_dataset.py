from manipulate import *
from globals import IMG_HEIGHT,IMG_WIDTH

class KittiDataset(object):
  """ A class for interacting with the KITTI dataset. """

  def __init__(
    self,image_dir,point_dir,calib_dir,label_dir,index_filename=None,
    is_training=True,is_raw=False,difficulty=-100,num_classes=8
  ):
    """ contructor
    
    @param image_dir[str]:       images path
    @param point_dir[str]:       point cloud data path
    @param calib_dir[str]:       calibration matrices path
    @param label_dir[str]:       label path
    @param index_filename[str]:  index file path
    @param is_training[bool]:    if True, dataset is training dataset
    @param is_raw[bool]:         if True, dataset is raw dataset, in which case there is no calibration file
    @param difficulty[int]:      difficulty level defines min box sizes
    @param num_classes[int]:     number of classes in dataset
    """

    self.image_dir      = image_dir
    self.point_dir      = point_dir
    self.calib_dir      = calib_dir
    self.label_dir      = label_dir
    self.index_filename = index_filename
    self.istraining     = is_training
    self.is_raw         = is_raw
    self.num_classes    = num_classes
    self.difficulty     = difficulty
    self.max_img_heigh  = 376
    self.max_img_width  = 1242

    if index_filename:
      self.file_list = self.read_index_file(self.index_filename)
    else:
      self.file_list = self.get_file_list(self.image_dir)

    self.num_files      = len(self.file_list)
    
    self.veify_file_list(
      self.img_dir, self.point_dir, self.calib_dir, self.label_dir, self.file_list,is_training,is_raw
    )
    

  def __str__(self):
    """ Generate string representation of dataset """
    sum_str = (
      'Dataset Summary:\n'
      + '\t img_dir=%s\n' % self.image_dir
      + '\t point_dir=%s\n' % self.point_dir
      + '\t calib_dir=%s\n' % self.calib_dir
      + '\t label_dir=%s\n' % self.label_dir
      + '\t index_filename=%s\n' % self.index_filename
      + '\t Total number of samples: %d\n' % self.num_files # TODO: THIS VARIABLE IS NOT DEFINED ERROR IN CODE !!!
    )
    statistics = self.get_statistics()

    return sum_str + statistics
  

  def get_statistics(self):
    from matplotlib import pyplot as plt
    from models import nms
    """ Get stats of objects inside the dataset """
    x_dict          = defaultdict(list)
    y_dict          = defaultdict(list)
    z_dict          = defaultdict(list)
    h_dict          = defaultdict(list)
    w_dict          = defaultdict(list)
    l_dict          = defaultdict(list)
    view_angle_dict = defaultdict(list)
    yaw_dict        = defaultdict(list)

    for frame_idx in range(self.num_files):
      labels = self.get_label(frame_idx)
      for label in labels:
        if label['ymin'] > 0:
          if label['ymax']- label['ymin'] > 25: # TODO: What is this restriction for?
            object_name = label['name']
            h_dict[object_name].append(label['height'])   # height
            w_dict[object_name].append(label['width'])    # width
            l_dict[object_name].append(label['length'])   # length
            x_dict[object_name].append(label['x3d'])      # x coord
            y_dict[object_name].append(label['y3d'])      # y coord
            z_dict[object_name].append(label['z3d'])      # z coord
            view_angle_dict[object_name].append(          # view angle
              np.arctan(label['x3d']/label['z3d'])
            )
            yaw_dict[object_name].append(label['yaw'])
    
    # plot pedestrians as scatter plot
    plt.scatter(z_dict['Pedestrian'],np.array(l_dict['Pedestrian'])) # plot singular points representing pedestrians
    plt.title('Pedestrian Scatter Plot')
    plt.show()

    # compute statistics
    truncation_rates    = []  # TODO: what are truncation rates for?
    no_truncation_rates = []  # TODO: same here
    image_height        = []
    image_width         = []

    for frame_idx in range(self.num_files):
      labels  = self.get_label(frame_idx) # TODO: This might be the key to the num_files problem
      calib   = self.get_calib(frame_idx)
      image   = self.get_image(frame_idx)
      image_height.append(image.shape[0])
      image_width.append(image.shape[1])

      for label in labels:
        if label['name'] == 'Car':
          object_name = label['name']
          # too small
          if label['ymax'] - label['ymin'] < 25:                    # ignore objects with insufficient height
            h_dict['ignored_by_height'].append(label['height'])
            w_dict['ignored_by_height'].append(label['width'])
            l_dict['ignored_by_height'].append(label['length'])
            x_dict['ignored_by_height'].append(label['x3d'])
            y_dict['ignored_by_height'].append(label['y3d'])
            z_dict['ignored_by_height'].append(label['z3d'])
            view_angle_dict['ignored_by_height'].append(
              np.arctan(label['x3d']/label['z3d'])
            )
            yaw_dict['ignored_by_height'].append(label['yaw'])
          if label['truncation'] > 0.5:                             # ignore objects that were truncated too much
            h_dict['ignored_by_truncation'].append(label['height'])
            w_dict['ignored_by_truncation'].append(label['width'])
            l_dict['ignored_by_truncation'].append(label['length'])
            x_dict['ignored_by_truncation'].append(label['x3d'])
            y_dict['ignored_by_truncation'].append(label['y3d'])
            z_dict['ignored_by_truncation'].append(label['z3d'])
            view_angle_dict['ignored_by_truncation'].append(
              np.arctan(label['x3d']/label['z3d'])
            )
            yaw_dict['ignored_by_truncation'].append(label['yaw'])

          detection_boxes_3d = np.array(                            # bounding boxes as array
            [[label['x3d'],label['y3d'],label['z3d'],
              label['length'],label['height'],label['width'],label['yaw']]]
          )
          detection_boxes_3d_corners = nms.boxes_3d_to_corners(detection_boxes_3d)  # get corners of bounding box
          corners_cam_points  = Points(xyz=detection_boxes_3d_corners[0],attr=None) # convert to Points object
          corners_img_points  = self.cam_points_to_image(corners_cam_points,calib)  # project to image plane
          corners_xy          = corners_img_points.xyz[:,:2]                        # get 2D coordinates
          xmax, ymax          = np.amax(corners_xy, axis=0)                         # get max values
          xmin, ymin          = np.amin(corners_xy, axis=0)                         # get min values
          clip_xmin           = np.clip(xmin,0)                                     # left clipping bound
          clip_ymin           = np.clip(ymin,0)                                     # bottom clipping bound                  
          clip_xmax           = np.clip(xmax,IMG_WIDTH)                             # right clipping bound
          clip_ymax           = np.clip(ymax,IMG_HEIGHT)                            # top clipping bound                    
          height              = clip_ymax - clip_ymin
          width               = clip_xmax - clip_xmin
          point_frame_size    = (ymax-ymin)*(xmax-xmin)
          truncation_rate    = 1.0 - height*width/point_frame_size                  # amount of datapoints outside frame

          # categorize labels by truncation rate
          if label['truncation'] > 0.5: truncation_rates.append(truncation_rate)    # ignore objects that were truncated too much
          else: no_truncation_rates.append(truncation_rate)   

          if label['occlusion'] > 2: # TODO: What the fuck is the occlusion rate!!! ???
            h_dict['ignored_by_occlusion'].append(label['height'])
            w_dict['ignored_by_occlusion'].append(label['width'])
            l_dict['ignored_by_occlusion'].append(label['length'])
            x_dict['ignored_by_occlusion'].append(label['x3d'])
            y_dict['ignored_by_occlusion'].append(label['y3d'])
            z_dict['ignored_by_occlusion'].append(label['z3d'])
            view_angle_dict['ignored_by_occlusion'].append( np.arctan(label['x3d']/label['z3d']) )
            yaw_dict['ignored_by_collection'].append(label['yaw'])

    stats = ""

    for object_name in h_dict:
      print(object_name + "l="+str(np.histogram(l_dict[object_name], 10, density=True)))
      if len(h_dict[object_name]):
        stats += (
          str(len(h_dict[object_name])) + " " + str(object_name)
          + " "
          + "mh=" + str(np.min(h_dict[object_name])) + " "
                  + str(np.median(h_dict[object_name])) + " "
                  + str(np.max(h_dict[object_name])) + "; "
          + "mw=" + str(np.min(w_dict[object_name])) + " "
                  + str(np.median(w_dict[object_name])) + " "
                  + str(np.max(w_dict[object_name])) + "; "
          + "ml=" + str(np.min(l_dict[object_name])) + " "
                  + str(np.median(l_dict[object_name])) + " "
                  + str(np.max(l_dict[object_name])) + "; "
          + "mx=" + str(np.min(x_dict[object_name])) + " "
                  + str(np.median(x_dict[object_name])) + " "
                  + str(np.max(x_dict[object_name])) + "; "
          + "my=" + str(np.min(y_dict[object_name])) + " "
                  + str(np.median(y_dict[object_name])) + " "
                  + str(np.max(y_dict[object_name])) + "; "
          + "mz=" + str(np.min(z_dict[object_name])) + " "
                  + str(np.median(z_dict[object_name])) + " "
                  + str(np.max(z_dict[object_name])) + "; "
          + "mA=" + str(np.min(view_angle_dict[object_name]))
          + " "
                  + str(np.median(view_angle_dict[object_name]))
          + " "
                + str(np.max(view_angle_dict[object_name])) + "; "
          + "mY=" + str(np.min(yaw_dict[object_name])) + " "
                  + str(np.median(yaw_dict[object_name])) + " "
                  + str(np.max(yaw_dict[object_name])) + "; "
          + "image_height" + str(np.min(image_height)) + " "
          + str(np.max(image_height)) +" "
          + "image_width" + str(np.min(image_width)) + " "
          + str(np.max(image_width)) + ";"
          "\n"
        )
        
    return stats


  def cam_points_to_image(self,points,calib): # TODO: homogenous coordinates used for projective transformations
    """ project Points onto image plane

    @param points[array(Nx3)]: 3D points in camera coordinates
    @param calib[dict]:         calibration matrix

    @return points[array(Nx2)]: 2D points in image coordinates
    """

    cam_points_xyz1 = np.hstack(                            # convert too homogenous coordinates
      [points.xyz,np.ones([points.xyz.shape[0],1])]
    )
    img_points_xyz  = np.matmul(                            # project points to homogenous image plane                           
      cam_points_xyz1, np.transpose(calib['cam_to_image']) 
    )
    img_points_xy1  = img_points_xyz/img_points_xyz[:,[2]]  # convert back to inhomogenous coordinates
    img_points = Points(img_points_xy1,points.attr)
    return img_points  


  def get_file_list(self, image_dir):
    """returns all file names inf image_dir

    @param image_dir[str]:  a string of path to the image folder.  
    @return:                a list of filenames.
    """

    file_list = [f.split('.')[0] for f in os.listdir(image_dir) if isfile(join(image_dir, f))]
    file_list.sort()
    return file_list
  
  def verify_file_list(self, image_dir, pc_dir, label_dir, calib_dir, file_list,is_training, is_raw):
    """Verify the files in file_list exist.

    @param: image_dir[str]:   path to image folder.
    @param: pc_dir[str]:      path to point cloud data folder.
    @param: label_dir[str]:   path to label folder.
    @param: calib_dir[str]:   path to  calibration folder.
    @param: file_list[list]:  filenames.
    @param: is_training[bool]:  if False, label_dir is not verified, as test dataset has no labels
    @param: is_raw[bool]:       if True, calib_dir is not verified, as raw dataset has no calibration files.

    @raise: assertion error when file in file_list is not complete.
    """

    for f in file_list:
      image_file  = join(image_dir, f)+'.png'
      pc_file     = join(pc_dir, f)+'.bin'
      label_file  = join(label_dir, f)+'.txt'
      calib_file  = join(calib_dir, f)+'.txt'

      assert isfile(image_file), "Image file %s does not exist" % image_file
      assert isfile(pc_file), "Point-Cloud file %s does not exist" % pc_file
      if not is_raw:  assert isfile(calib_file), "Calibration file %s does not exist" % calib_file
      if is_training: assert isfile(label_file),"Label %s does not exist" % label_file

  
  def read_index_file(self, index_filename):
    """Read an index file containing the filenames.

    @param index_filename[str]: a string containing the path to an index file.
    Returns: a list of filenames.
    """

    file_list = []
    with open(index_filename, 'r') as f:
        for line in f:
            file_list.append(line.rstrip('\n').split('.')[0])
    return file_list  