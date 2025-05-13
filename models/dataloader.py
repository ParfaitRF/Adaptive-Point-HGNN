from typing import Tuple,List,Callable
import numpy as np

from models.box_encoders_decoders import (
  get_box_encoding_decoding_fn,get_encoding_len
)
from models.graph_gen import get_graph_generate_fn
from data import preprocess
from data.kitti_dataset import KittiDataset
from torch.utils.data import Dataset,DataLoader


def batch_data(batch_list:List) -> Tuple[
  List[np.ndarray],List[np.ndarray],List[np.ndarray],List[np.ndarray],
  List[np.ndarray],List[np.ndarray]
]:
  N_input_v, N_vertex_idx_list, N_keypoint_indices_list, N_edges_list,\
  N_cls_labels, N_encoded_boxes, N_valid_boxes = zip(*batch_list)
  batch_size  = len(batch_list)
  level_num   = len(N_vertex_idx_list[0])
  batched_keypoint_indices_list = []
  batched_edges_list            = []

  for level_idx in range(level_num-1):                                          # iterate over layers
    centers         = []
    vertices        = []
    point_counter   = 0
    center_counter  = 0

    for batch_idx in range(batch_size):                                         # iterate over all underlying pcds              
      centers.append(
        N_keypoint_indices_list[batch_idx][level_idx]+point_counter
      )
      vertices.append(np.hstack([
        N_edges_list[batch_idx][level_idx][:,[0]]+point_counter,
        N_edges_list[batch_idx][level_idx][:,[1]]+center_counter
      ]))
      point_counter   += N_vertex_idx_list[batch_idx][level_idx].shape[0]
      center_counter  += \
        N_keypoint_indices_list[batch_idx][level_idx].shape[0]
    batched_keypoint_indices_list.append(np.vstack(centers))
    batched_edges_list.append(np.vstack(vertices))
  batched_vertex_coord_list = []

  for level_idx in range(level_num):
    points  = []
    for batch_idx in range(batch_size):
      points.append(N_vertex_idx_list[batch_idx][level_idx])
    batched_vertex_coord_list.append(np.vstack(points))

  batched_input_v         = np.vstack(N_input_v)
  batched_cls_labels      = np.vstack(N_cls_labels)
  batched_encoded_boxes   = np.vstack(N_encoded_boxes)
  batched_valid_boxes     = np.vstack(N_valid_boxes)

  return (batched_input_v, batched_vertex_coord_list,
      batched_keypoint_indices_list, batched_edges_list, batched_cls_labels,
      batched_encoded_boxes, batched_valid_boxes)


class CustomKittiDataset(Dataset,KittiDataset):
  def __init__(
    self,config:dict,train_config:dict, is_training:bool=True, is_raw:bool=False, 
    difficulty:int=0,num_classes:int=8
  ):
    Dataset.__init__(self)                                                      # initialize dataset class
    KittiDataset.__init__(                                                      # initialize kitti dataset class
      self,is_training=is_training,is_raw=is_raw,difficulty=difficulty,
      num_classes=num_classes
    )
    self.config = config                                                        # set configuration
    self.train_config = train_config                                            # set training configuration

    self.box_encoding_fn,self.box_decoding_fn = get_box_encoding_decoding_fn(   # get box encoding and decoding functions 
      self.config['box_encoding_method']
    )
    

  def __len__(self):                                                            # get dataset length                     
    return self.num_files
  

  def __getitem__(self,frame_idx:int) -> Tuple[
    np.ndarray,list[np.ndarray],list[np.ndarray],np.ndarray,np.ndarray,
    np.ndarray[bool]
  ]:
    """ Fetch data from the dataset.
    Parameters
    ----------
    frame_idx: int
      index of the frame to fetch

    Returns
    -------
      data: dictionary containing the data for the frame
    """

    cam_rgb_points = self.get_cam_points_in_image_with_rgb(
      frame_idx,self.config['downsample_voxel_size']
    )
    labels = self.get_label(frame_idx)

    if 'crop_aug' in self.train_config:                                         # needs to be added to config
      cam_rgb_points, labels = self.crop_aug(                                   # add cropped objects into frame
        cam_rgb_points,labels,
        sample_rate=self.train_config['crop_aug']['sample_rate'],
        parser_kwargs=self.train_config['crop_aug']['parser_kwargs']
      )
    
    aug_fn = preprocess.get_data_aug(self.train_config['data_aug_configs'])
    cam_rgb_points, labels = aug_fn(cam_rgb_points, labels)                     # data augmentation
    graph_generate_fn = get_graph_generate_fn(self.config['graph_gen_method'])
    vertex_idx_list, edges_list = graph_generate_fn(
      cam_rgb_points.xyz, **self.config['graph_gen_kwargs']
    )
    # VERTEX FEATURES
    if self.config['input_features'] == 'irgb':                                 # reflectivity + rgb                                                                     
      input_v = cam_rgb_points.attr                                             
    elif self.config['input_features'] == '0rgb':                               # zero + rgb           
      input_v = np.hstack([
        np.zeros((cam_rgb_points.attr.shape[0], 1)),cam_rgb_points.attr[:, 1:]
      ])
    elif self.config['input_features'] == '0000':                               # zeros         
      input_v = np.zeros_like(cam_rgb_points.attr)
    elif self.config['input_features'] == 'i000':                               # reflectivity + zeros           
      input_v = np.hstack([
        cam_rgb_points.attr[:, [0]],np.zeros((cam_rgb_points.attr.shape[0], 3))
      ])
    elif self.config['input_features'] == 'i':                                  # reflectivity        
      input_v = cam_rgb_points.attr[:, [0]]
    elif self.config['input_features'] == '0':                                  # zero            
      input_v = np.zeros((cam_rgb_points.attr.shape[0], 1))
    input_v = input_v.astype(np.float64)                                        # convert input vertex features to float32

    last_layer_points_xyz = cam_rgb_points.xyz[vertex_idx_list[-1]]             # get last layer points 
                        
    cls_labels, boxes_3d, valid_boxes = self.assign_classaware_label_to_points( # assign labels to points   
      labels,
      last_layer_points_xyz,
      expend_factor=self.train_config.get('expend_factor', (1.0, 1.0, 1.0)),
      label_method=self.config['label_method']
    )

    encoded_boxes   = self.box_encoding_fn(                                     # encode bounding boxes
      cls_labels, last_layer_points_xyz,boxes_3d, self
    )

    return (
      input_v,vertex_idx_list,edges_list,cls_labels,encoded_boxes,valid_boxes
    )
  
