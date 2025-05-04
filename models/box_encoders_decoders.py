""" This file contains functions to encode bounding boxes """

import numpy as np
from globals import OBJECTS_DICT

# median_object_size_map = {
#   'Cyclist': (1.76, 1.75, 0.6),
#   'Van': (4.98, 2.13, 1.88),
#   'Tram': (14.66, 3.61, 2.6),
#   'Car': (3.88, 1.5, 1.63),
#   'Misc': (2.52, 1.65, 1.51),
#   'Pedestrian': (0.88, 1.77, 0.65),
#   'Truck': (10.81, 3.34, 2.63),
#   'Person_sitting': (0.75, 1.26, 0.59),
#   # 'DontCare': (-1.0, -1.0, -1.0)
# }


def classaware_all_class_box_encoding(cls_labels, points_xyz, boxes_3d,kitti_dataset):
  r""" The bounding box is encoded with the vertex coordinates as follows:

  .. math::

    \begin{equation}
      \begin{split}
        \delta_x      & = \frac{x-x_v}{l_m}, \quad \delta_y = 
        \frac{y-y_v}{h_m}, \quad \delta_z = \frac{z-z_v}{w_m}       \\
        \delta_l      & = \log(\frac{l}{l_m}), \quad \delta_h = 
        \log(\frac{h}{h_m}), \quad \delta_w = \log(\frac{w}{w_m}) \\
        \delta_\theta & = \frac{\theta-\theta_0}{\theta_m}
      \end{split}
    \end{equation}

  Parameters
    ----------
    cls_labels: np.ndarray
      The class labels of the points in the point cloud.
    points_xyz: np.ndarray
      The xyz coordinates of the points in the point cloud.
    encoded_boxes: np.ndarray
      The decoded bounding boxes.
    label_map: dict
      The mapping of class names to class labels.
    
    Returns
    -------
    encoded_boxes_3d: np.ndarray
      The encoded bounding boxes.
  """
  
  encoded_boxes_3d  = np.copy(boxes_3d)
  num_classes       = boxes_3d.shape[1]
  points_xyz        = np.expand_dims(points_xyz, axis=1)
  points_xyz        = np.tile(points_xyz, (1, num_classes, 1))
  encoded_boxes_3d[:, :, 0] -= points_xyz[:, :, 0]
  encoded_boxes_3d[:, :, 1] -= points_xyz[:, :, 1]
  encoded_boxes_3d[:, :, 2] -= points_xyz[:, :, 2]

  median_object_size_map = {
    obj : (
      np.median(kitti_dataset.l_dict[obj]),
      np.median(kitti_dataset.h_dict[obj]),
      np.median(kitti_dataset.w_dict[obj])
    )
    for obj in OBJECTS_DICT
  }

  for cls_name in OBJECTS_DICT:
    if cls_name in list(OBJECTS_DICT.keys())[0] or\
    cls_name in list(OBJECTS_DICT.keys())[-1] :  continue

    cls_label   = OBJECTS_DICT[cls_name]
    l, h, w     = median_object_size_map[cls_name]
    mask        = cls_labels[:, 0] == cls_label

    encoded_boxes_3d[mask, 0, 0] /= l
    encoded_boxes_3d[mask, 0, 1] /= h
    encoded_boxes_3d[mask, 0, 2] /= w
    encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/l)
    encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/h)
    encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/w)
    encoded_boxes_3d[mask, 0, 6] = boxes_3d[mask, 0, 6]/(np.pi/4)
    # vertical
    mask = cls_labels[:, 0] == (cls_label+1)
    encoded_boxes_3d[mask, 0, 0] /= l
    encoded_boxes_3d[mask, 0, 1] /= h
    encoded_boxes_3d[mask, 0, 2] /= w
    encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/l)
    encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/h)
    encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/w)

    encoded_boxes_3d[mask, 0, 6] -= (np.pi/2)
    encoded_boxes_3d[mask, 0, 6] /= (np.pi/4)
      
  return encoded_boxes_3d


def classaware_all_class_box_decoding(
  cls_labels, points_xyz, encoded_boxes,kitti_dataset
):
  r""" The bounding box is dencoded accoring to the inverse of the following:

  .. math::

    \begin{equation}
      \begin{split}
        \delta_x      & = \frac{x-x_v}{l_m}, \quad \delta_y = 
        \frac{y-y_v}{h_m}, \quad \delta_z = \frac{z-z_v}{w_m}       \\
        \delta_l      & = \log(\frac{l}{l_m}), \quad \delta_h = 
        \log(\frac{h}{h_m}), \quad \delta_w = \log(\frac{w}{w_m}) \\
        \delta_\theta & = \frac{\theta-\theta_0}{\theta_m}
      \end{split}
    \end{equation}

    Parameters
    ----------
    cls_labels: np.ndarray
      The class labels of the points in the point cloud.
    points_xyz: np.ndarray
      The xyz coordinates of the points in the point cloud.
    encoded_boxes: np.ndarray
      The encoded bounding boxes.
    label_map: dict
      The mapping of class names to class labels.
    
    Returns
    -------
    decoded_boxes_3d: np.ndarray
      The decoded bounding boxes.
  """
  decoded_boxes_3d = np.copy(encoded_boxes)

  median_object_size_map = {
   obj : (
      np.median(kitti_dataset.l_dict[obj]),
      np.median(kitti_dataset.h_dict[obj]),
      np.median(kitti_dataset.w_dict[obj])
   )
   for obj in OBJECTS_DICT.keys()
  }

  for cls_name in OBJECTS_DICT:
    if cls_name in list(OBJECTS_DICT.keys())[0] or\
    cls_name in list(OBJECTS_DICT.keys())[-1] :  continue

    cls_label = OBJECTS_DICT[cls_name]
    l, h, w   = median_object_size_map[cls_name]
    # Car horizontal
    mask      = cls_labels[:, 0] == cls_label
    decoded_boxes_3d[mask, 0, 0] *= l
    decoded_boxes_3d[mask, 0, 1] *= h
    decoded_boxes_3d[mask, 0, 2] *= w
    decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*l
    decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*h
    decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*w
    decoded_boxes_3d[mask, 0, 6] = encoded_boxes[mask, 0, 6]*(np.pi/4)
    # Car vertical
    mask = cls_labels[:, 0] == (cls_label+1)
    decoded_boxes_3d[mask, 0, 0] *= l
    decoded_boxes_3d[mask, 0, 1] *= h
    decoded_boxes_3d[mask, 0, 2] *= w
    decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*l
    decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*h
    decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*w
    decoded_boxes_3d[mask, 0, 6] = (encoded_boxes[mask, 0, 6])*(np.pi/4)+np.pi/2
  # offset
  num_classes = encoded_boxes.shape[1]
  points_xyz  = np.expand_dims(points_xyz, axis=1)
  points_xyz  = np.tile(points_xyz, (1, num_classes, 1))
  decoded_boxes_3d[:, :, 0] += points_xyz[:, :, 0]
  decoded_boxes_3d[:, :, 1] += points_xyz[:, :, 1]
  decoded_boxes_3d[:, :, 2] += points_xyz[:, :, 2]

  return decoded_boxes_3d


def classaware_all_class_box_canonical_encoding(
  cls_labels, points_xyz,boxes_3d,kitti_dataset):
  
  boxes_3d = np.copy(boxes_3d)
  num_classes = boxes_3d.shape[1]
  points_xyz = np.expand_dims(points_xyz, axis=1)
  points_xyz = np.tile(points_xyz, (1, num_classes, 1))
  boxes_3d[:, :, 0] = boxes_3d[:, :, 0] - points_xyz[:, :, 0]
  boxes_3d[:, :, 1] = boxes_3d[:, :, 1] - points_xyz[:, :, 1]
  boxes_3d[:, :, 2] = boxes_3d[:, :, 2] - points_xyz[:, :, 2]
  encoded_boxes_3d = np.copy(boxes_3d)

  median_object_size_map = {
   obj : (
      np.median(kitti_dataset.l_dict[obj]),
      np.median(kitti_dataset.h_dict[obj]),
      np.median(kitti_dataset.w_dict[obj])
   )
   for obj in OBJECTS_DICT.keys()
  }

  for cls_name in OBJECTS_DICT:
    if cls_name in list(OBJECTS_DICT.keys())[0] or\
    cls_name in list(OBJECTS_DICT.keys())[-1] :  continue

    cls_label = OBJECTS_DICT[cls_name]
    l, h, w   = median_object_size_map[cls_name]
    mask      = cls_labels[:, 0] == cls_label

    encoded_boxes_3d[mask, 0, 0] = (
      boxes_3d[mask, 0, 0]*np.cos(boxes_3d[mask, 0, 6]) \
      -boxes_3d[mask, 0, 2]*np.sin(boxes_3d[mask, 0, 6]))/l
    encoded_boxes_3d[mask, 0, 1] = boxes_3d[mask, 0, 1]/h
    encoded_boxes_3d[mask, 0, 2] = (
      boxes_3d[mask, 0, 0]*np.sin(boxes_3d[mask, 0, 6]) \
      +boxes_3d[mask, 0, 2]*np.cos(boxes_3d[mask, 0, 6]))/w
    encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/l)
    encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/h)
    encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/w)
    encoded_boxes_3d[mask, 0, 6] = boxes_3d[mask, 0, 6]/(np.pi/4)

    # vertical
    mask = cls_labels[:, 0] == (cls_label+1)
    encoded_boxes_3d[mask, 0, 0] = (
      boxes_3d[mask, 0, 0]*np.cos(boxes_3d[mask, 0, 6]-np.pi/2) \
      -boxes_3d[mask, 0, 2]*np.sin(boxes_3d[mask, 0, 6]-np.pi/2))/w
    encoded_boxes_3d[mask, 0, 1] = boxes_3d[mask, 0, 1]/h
    encoded_boxes_3d[mask, 0, 2] = (
      boxes_3d[mask, 0, 0]*np.sin(boxes_3d[mask, 0, 6]-np.pi/2) \
      +boxes_3d[mask, 0, 2]*np.cos(boxes_3d[mask, 0, 6]-np.pi/2))/l
    encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/l)
    encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/h)
    encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/w)
    encoded_boxes_3d[mask, 0, 6] = (boxes_3d[mask, 0, 6]-np.pi/2)/(np.pi/4)
           
  return encoded_boxes_3d


def classaware_all_class_box_canonical_decoding(
  cls_labels, points_xyz,encoded_boxes, kitti_dataset):

  decoded_boxes_3d = np.copy(encoded_boxes)
  median_object_size_map = {
   obj : (
    np.median(kitti_dataset.l_dict[obj]),
    np.median(kitti_dataset.h_dict[obj]),
    np.median(kitti_dataset.w_dict[obj])
   )
   for obj in OBJECTS_DICT.keys()
  }


  for cls_name in OBJECTS_DICT:
    if cls_name in list(OBJECTS_DICT.keys())[0] or\
    cls_name in list(OBJECTS_DICT.keys())[-1] :  continue
    cls_label = OBJECTS_DICT[cls_name]
    l, h, w   = median_object_size_map[cls_name]
    # Car horizontal
    mask      = cls_labels[:, 0] == cls_label

    decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*l*np.cos(
      encoded_boxes[mask, 0, 6]*(np.pi/4))\
      +encoded_boxes[mask, 0, 2]*w*np.sin(
      encoded_boxes[mask, 0, 6]*(np.pi/4))
    decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*h
    decoded_boxes_3d[mask, 0, 2] = -encoded_boxes[mask, 0, 0]*l*np.sin(
      encoded_boxes[mask, 0, 6]*(np.pi/4))\
      +encoded_boxes[mask, 0, 2]*w*np.cos(
      encoded_boxes[mask, 0, 6]*(np.pi/4))

    decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*l
    decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*h
    decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*w
    decoded_boxes_3d[mask, 0, 6] = encoded_boxes[mask, 0, 6]*(np.pi/4)

    # Car vertical
    mask = cls_labels[:, 0] == (cls_label+1)
    decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*w*np.cos(
      encoded_boxes[mask, 0, 6]*(np.pi/4))\
      +encoded_boxes[mask, 0, 2]*l*np.sin(
      encoded_boxes[mask, 0, 6]*(np.pi/4))
    decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*h
    decoded_boxes_3d[mask, 0, 2] = -encoded_boxes[mask, 0, 0]*w*np.sin(
      encoded_boxes[mask, 0, 6]*(np.pi/4))\
      +encoded_boxes[mask, 0, 2]*l*np.cos(
      encoded_boxes[mask, 0, 6]*(np.pi/4))

    decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*l
    decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*h
    decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*w
    decoded_boxes_3d[mask, 0, 6] = (
      encoded_boxes[mask, 0, 6])*(np.pi*0.25)+np.pi/2
  # offset
  num_classes = encoded_boxes.shape[1]
  points_xyz  = np.expand_dims(points_xyz, axis=1)
  points_xyz  = np.tile(points_xyz, (1, num_classes, 1))
  decoded_boxes_3d[:, :, 0] += points_xyz[:, :, 0]
  decoded_boxes_3d[:, :, 1] += points_xyz[:, :, 1]
  decoded_boxes_3d[:, :, 2] += points_xyz[:, :, 2]
  
  return decoded_boxes_3d


def get_box_encoding_fn(encoding_method_name):
  encoding_method_dict = {
    'classaware_all_class_box':classaware_all_class_box_encoding,
    'classaware_all_class_box_canonical':
      classaware_all_class_box_canonical_encoding,
  }
  return encoding_method_dict[encoding_method_name]


def get_box_decoding_fn(encoding_method_name):
  decoding_method_dict = {
    'classaware_all_class_box': classaware_all_class_box_decoding,
    'classaware_all_class_box_canonical':
      classaware_all_class_box_canonical_decoding,
  }
  return decoding_method_dict[encoding_method_name]


def get_encoding_len(encoding_method_name):
  encoding_len_dict = {
    _key:7 for _key in [
      'classaware_all_class_box',
      'classaware_all_class_box_canonical'
    ]
  }
  return encoding_len_dict[encoding_method_name]


def test_encoder_decoder(encoding_type,kitti_dataset):
  encoding_fn = get_box_encoding_fn(encoding_type)
  decoding_fn = get_box_decoding_fn(encoding_type)

  num_samples = 10000
  cls_labels  = np.random.choice(
    list(OBJECTS_DICT.values()), (num_samples, 1))
  points_xyz  = np.random.random((num_samples, 3))*10
  boxes_3d    = np.random.random((num_samples, 1, 7))*10
  boxes_3d[:, :, 3:6] = np.absolute(boxes_3d[:, :, 3:6])
  
  encoded_boxes = encoding_fn(cls_labels, points_xyz, boxes_3d, kitti_dataset)
  decoded_boxes = decoding_fn(cls_labels, points_xyz, encoded_boxes, kitti_dataset)
  
  assert np.isclose(decoded_boxes, boxes_3d).all()
