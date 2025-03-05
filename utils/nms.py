""" contains all functionalities needed for non-maximum suppression """
from typing import Callable
import cv2
import numpy as np
from shapely.geometry import Polygon                                            # A polygon is a 2D object

from globals import COLOR1
from data.transformations import boxes_3d_to_corners

    
def overlapped_boxes_3d(single_box:np.array, box_list:list):
  """ Computes the overlap between a single box and multiple query boxes

  @param single_box:  a single bounding box
  @param box_list:    list of bounding boxes 

  @return:            list of overlaps
  """

  x0_max, y0_max, z0_max = np.max(single_box, axis=0)                           # get min and max values of single box
  x0_min, y0_min, z0_min = np.min(single_box, axis=0)
  overlap = np.zeros(len(box_list))                                             # initialize overlap list

  for i, box in enumerate(box_list):
    x_max, y_max, z_max = np.max(box, axis=0)                                   # get min and max of query box                    
    x_min, y_min, z_min = np.min(box, axis=0)

    if x0_max < x_min or x0_min > x_max:                                        # caces of no overlap              
      overlap[i] = 0
      continue
    if y0_max < y_min or y0_min > y_max:
      overlap[i] = 0
      continue
    if z0_max < z_min or z0_min > z_max:
      overlap[i] = 0
      continue
    
    x_draw_min  = min(x0_min, x_min)                                            # define frame dimensions
    x_draw_max  = max(x0_max, x_max)
    z_draw_min  = min(z0_min, z_min)
    z_draw_max  = max(z0_max, z_max)
    offset      = np.array([x_draw_min, z_draw_min])                            # define offset
    buf1        = np.zeros((z_draw_max-z_draw_min, x_draw_max-x_draw_min),      # create buffes
                           dtype=np.int32)
    buf2        = np.zeros_like(buf1)
    cv2.fillPoly(buf1, [single_box[:4, [0,2]]-offset], color=COLOR1)            # fill polygon representation in buffer
    cv2.fillPoly(buf2, [box[:4, [0,2]]-offset], color=COLOR1)                   # fill polygon representation in buffer
    shared_area = cv2.countNonZero(buf1*buf2)                                   # intersection area
    area1 = cv2.countNonZero(buf1)                                              # compute overlap...                 
    area2 = cv2.countNonZero(buf2)
    shared_y      = min(y_max, y0_max) - max(y_min, y0_min)
    intersection  = shared_y * shared_area
    union         = (y_max-y_min) * area2 + (y0_max-y0_min) * area1
    overlap[i]    = np.float32(intersection) / (union - intersection)

  return overlap


def overlapped_boxes_3d_fast_poly(single_box:np.array, box_list:list):
  """ Computes the overlap between a single box and multiple query boxes

  @param single_box: a single bounding box
  @param box_list:   list of bounding boxes 

  @return:            list of overlaps
  """

  if box_list.shape == (0,): return np.zeros(0)                                 # return empty list if no boxes

  single_box_max_corner = np.max(single_box, axis=0)                            # get max values of all dimensions
  single_box_min_corner = np.min(single_box, axis=0)                            # get min values of all dimensions                    
  x0_max, y0_max, z0_max = single_box_max_corner                                # get coordinate values of max/min
  x0_min, y0_min, z0_min = single_box_min_corner
  max_corner    = np.max(box_list, axis=1)                                      # get max and min lits for list of query boxes
  min_corner    = np.min(box_list, axis=1)
  overlap       = np.zeros(len(box_list))                                       # intialize overlap list
  non_overlap_mask =  np.logical_or(single_box_max_corner < min_corner,         # mask indicating which boxes overlap
                                    single_box_min_corner > max_corner)
  overlap_mask  = np.logical_not(np.any(non_overlap_mask, axis=1))
  p1    = Polygon(single_box[:4, [0,2]])
  area1 = p1.area                                                               # get single box area

  for i,box in enumerate(box_list):
    if overlap_mask[i]:
      x_max, y_max, z_max = max_corner[i]
      x_min, y_min, z_min = min_corner[i]
      p2    =  Polygon(box[:4, [0,2]])
      shared_area   = p1.intersection(p2).area                                  # get 2D overlap area   
      area2 = p2.area
      shared_y      = min(y_max, y0_max) - max(y_min, y0_min)
      intersection  = shared_y * shared_area                                    # get 3D overlap volume
      union = (y_max-y_min) * area2 + (y0_max-y0_min) * area1
      overlap[i]    = np.float32(intersection) / (union - intersection)         # IoU

  return overlap


def bboxes_sort(
    classes:list, scores:list, bboxes:list, top_k:int=400, attributes:list=None):
  """ Sorts bounding boxes by decreasing order of scores and keep only the top k

  @param classes:    list of classes
  @param scores:     list of scores
  @param bboxes:     list of bounding boxes
  @param top_k:      number of bounding boxes to keep
  @param attributes: list of attributes
  """
  if top_k <= 0: raise Exception('top_k should be positive')

  idxes   = np.argsort(-scores)
  classes = classes[idxes]
  scores  = scores[idxes]
  bboxes  = bboxes[idxes]

  if attributes is not None:  attributes = attributes[idxes]

  if len(idxes) > top_k:
    classes = classes[:top_k]
    scores  = scores[:top_k]
    bboxes  = bboxes[:top_k]

    if attributes is not None:  attributes = attributes[:top_k]
        
  return classes, scores, bboxes, attributes


def bboxes_nms(
    classes:list, scores:list, bboxes:list, nms_threshold:float=0.45,
    overlapped_fn:Callable=overlapped_boxes_3d_fast_poly,appr_factor:float=10.0, 
    attributes:list=None):
  """ Applies non-maximum supression to bounding boxes

  @param classes:        list of classes
  @param scores:         list of scores
  @param bboxes:         list of bounding boxes
  @param nms_threshold:  IoU threshold
  @param overlapped_fn:  function to compute overlap
  @param appr_factor:    approximation factor
  @param attributes:     list of attributes	

  @return:  list of classes, scores, bounding boxes, and attributes of boxes 
            that passed NMS
  """

  boxes_corners = boxes_3d_to_corners(bboxes)                                   # convert to camera coordinates
  boxes_corners = np.int32(boxes_corners*appr_factor)                           # apply approximation factor          
  keep_bboxes   = np.ones(scores.shape, dtype=np.bool)                          # boolean mask for boxes to be kept

  for i in range(scores.size-1):
    if keep_bboxes[i]:
      overlap       = overlapped_fn(boxes_corners[i], boxes_corners[(i+1):])    # compute overlap
      keep_overlap  = np.logical_or(                                            # keep those outside overlap threshold or diff class
        overlap <= nms_threshold, classes[(i+1):] != classes[i])
      keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)   # update mask
          
  idxes   = np.where(keep_bboxes)
  classes = classes[idxes]
  scores  = scores[idxes]
  bboxes  = bboxes[idxes]
  
  if attributes is not None: attributes = attributes[idxes]

  return classes, scores, bboxes, attributes


def bboxes_nms_uncertainty(
    classes:list, scores:list, bboxes:np.array,scores_threshold:float=0.25,
    nms_threshold:float=0.45,overlapped_fn:Callable=overlapped_boxes_3d,
    appr_factor:float=10.0,attributes:list=None):
  """ Applies non-maximum supression to bounding boxes with uncertainty 
  
  @param classes:          list of classes
  @param scores:           list of scores
  @param bboxes:           list of bounding boxes
  @param scores_threshold: threshold for scores
  @param nms_threshold:    IoU threshold
  @param overlapped_fn:    function to compute overlap
  @param appr_factor:      approximation factor
  @param attributes:       list of attributes

  @return:  list of classes, scores, bounding boxes, and attributes of boxes 
            that passed NMS
  """

  boxes_corners = boxes_3d_to_corners(bboxes)                                   # convert to camera coordinates
  keep_bboxes   = np.ones(scores.shape, dtype=np.bool)

  for i in range(scores.size-1):
    if keep_bboxes[i]:
      valid   = keep_bboxes[(i+1):]                                             # boxes valid for iteration             
      overlap = overlapped_fn(boxes_corners[i], boxes_corners[(i+1):][valid])   # compute overlap with valid boxes
      remove_overlap      = np.logical_and(                                     # mask for boxes to be supressed 
        overlap > nms_threshold, classes[(i+1):][valid] == classes[i])
      overlaped_bboxes    = np.concatenate(                                     # boxes to be supressed                
        [bboxes[(i+1):][valid][remove_overlap], bboxes[[i]]], axis=0)
      boxes_mean    = np.median(overlaped_bboxes, axis=0)                       # get mean of overlapping boxes (median !!)
      bboxes[i][:]  = boxes_mean[:]                                             # update box with mean                       
      boxes_corners_mean  = boxes_3d_to_corners(                                # converts to camera coordinates (again??)
        np.expand_dims(boxes_mean, axis=0))
      boxes_mean_overlap  = overlapped_fn(                                      # ?? DONT CHANGE A RUNNING SYSTEM
        boxes_corners_mean[0],
        boxes_corners[(i+1):][valid][remove_overlap])
      scores[i] += np.sum(                                                      # update score with overlap                       
        scores[(i+1):][valid][remove_overlap]*boxes_mean_overlap)
      keep_bboxes[(i+1):][valid] = np.logical_not(remove_overlap)               # update mask

  idxes   = np.where(keep_bboxes)
  classes = classes[idxes]
  scores  = scores[idxes]
  bboxes  = bboxes[idxes]
  
  if attributes is not None:  attributes = attributes[idxes]

  return classes, scores, bboxes, attributes


def bboxes_nms_merge_only(
    classes:list, scores:list, bboxes:list,scores_threshold:float=0.25,
    nms_threshold:float=0.45,overlapped_fn:Callable=overlapped_boxes_3d,
    appr_factor:float=10.0,attributes:list=None):
  """ Applies non-maximum selection to bounding boxes.

  @param classes:          list of classes
  @param scores:           list of scores
  @param bboxes:           list of bounding boxes
  @param scores_threshold: threshold for scores
  @param nms_threshold:    IoU threshold
  @param overlapped_fn:    function to compute overlap
  @param appr_factor:      approximation factor
  @param attributes:       list of attributes

  @return:  list of classes, scores, bounding boxes, and attributes of boxes 
            that passed NMS
  """
  boxes_corners = boxes_3d_to_corners(bboxes)                                   # convert to camera coordinates
  keep_bboxes   = np.ones(scores.shape, dtype=np.bool)

  for i in range(scores.size-1):
    if keep_bboxes[i]:
      valid   = keep_bboxes[(i+1):]                                             # boxes valid for iteration       
      overlap = overlapped_fn(boxes_corners[i],boxes_corners[(i+1):][valid])    # compute overlap with valid boxes
      remove_overlap    = np.logical_and(overlap > nms_threshold,               # mask for boxes to be supressed                
        classes[(i+1):][valid] == classes[i])
      overlaped_bboxes  = np.concatenate(                                       # boxes to be supressed                   
        [bboxes[(i+1):][valid][remove_overlap], bboxes[[i]]], axis=0)
      boxes_mean    = np.median(overlaped_bboxes, axis=0)                       # get mean of overlapping boxes (median !!)
      bboxes[i][:]  = boxes_mean[:]                                             # update box with mean       
      keep_bboxes[(i+1):][valid] = np.logical_not(remove_overlap)               # update mask

  idxes   = np.where(keep_bboxes)
  classes = classes[idxes]
  scores  = scores[idxes]
  bboxes  = bboxes[idxes]

  if attributes is not None:  attributes = attributes[idxes]

  return classes, scores, bboxes, attributes


def bboxes_nms_score_only(
    classes:list, scores:list, bboxes:list,scores_threshold:float=0.25,
    nms_threshold:float=0.45,overlapped_fn:Callable=overlapped_boxes_3d,
    appr_factor:float=10.0,attributes:list=None):
  """ Applies non-maximum selection to bounding boxes.

  @param classes:          list of classes
  @param scores:           list of scores
  @param bboxes:           list of bounding boxes
  @param scores_threshold: threshold for scores
  @param nms_threshold:    IoU threshold
  @param overlapped_fn:    function to compute overlap
  @param appr_factor:      approximation factor 
  @param attributes:       list of attributes

  @return:  list of classes, scores, bounding boxes, and attributes of boxes 
            that passed NMS
  """
  boxes_corners = boxes_3d_to_corners(bboxes)                                   # convert to camera coordinates    
  keep_bboxes   = np.ones(scores.shape, dtype=np.bool)

  for i in range(scores.size-1):
    if keep_bboxes[i]:
      valid   = keep_bboxes[(i+1):]                                             # boxes valid for iteration
      overlap = overlapped_fn(boxes_corners[i],boxes_corners[(i+1):][valid])    # compute overlap with valid boxes
      remove_overlap      = np.logical_and(overlap > nms_threshold,             # mask for boxes to be supressed 
        classes[(i+1):][valid] == classes[i])
      overlaped_bboxes    = np.concatenate(                                     # boxes to be supressed                    
        [bboxes[(i+1):][valid][remove_overlap], bboxes[[i]]], axis=0)
      boxes_mean    = bboxes[i][:]                                              # get mean of overlapping boxes (median !!)
      bboxes[i][:]  = boxes_mean[:]                                             # update box with mean 
      boxes_corners_mean  = boxes_3d_to_corners(                                # update mask
        np.expand_dims(boxes_mean, axis=0))
      boxes_mean_overlap  = overlapped_fn(boxes_corners_mean[0],                # compute overlap between boxes and mean
        boxes_corners[(i+1):][valid][remove_overlap])
      scores[i] += np.sum(                                                      # update score with overlap
        scores[(i+1):][valid][remove_overlap]*boxes_mean_overlap)
      keep_bboxes[(i+1):][valid] = np.logical_not(remove_overlap)               # update mask        

  idxes   = np.where(keep_bboxes)
  classes = classes[idxes]
  scores  = scores[idxes]
  bboxes  = bboxes[idxes]

  if attributes is not None:  attributes = attributes[idxes]

  return classes, scores, bboxes, attributes


def nms_boxes_3d(
    class_labels:list, detection_boxes_3d:np.array,detection_scores:list,
    overlapped_thres:float=0.5,overlapped_fn:Callable=overlapped_boxes_3d,
    appr_factor:float=10.0,top_k:int=-1, attributes:list=None):
  """ Applies non-maximum selection to bounding boxes.
  
  @param class_labels:        list of classes
  @param detection_boxes_3d:  list of bounding boxes
  @param detection_scores:    list of scores
  @param overlapped_thres:    IoU threshold
  @param overlapped_fn:       function to compute overlap
  @param appr_factor:         approximation factor
  @param top_k:               number of bounding boxes to keep
  @param attributes:          list of attributes

  @return: filtered list of classes, scores, bounding boxes, and attributes
  """
  class_labels, detection_scores, detection_boxes_3d, attributes = bboxes_sort( # sort bounding by score
    class_labels, detection_scores, detection_boxes_3d, top_k=top_k,
    attributes=attributes)
  # non maximum supression
  class_labels, detection_scores, detection_boxes_3d, attributes = bboxes_nms(  # compute NMS
    class_labels, detection_scores, detection_boxes_3d,
    nms_threshold=overlapped_thres, overlapped_fn=overlapped_fn,
    appr_factor=appr_factor, attributes=attributes)
  
  return class_labels, detection_boxes_3d, detection_scores, attributes         # return filtered results


def nms_boxes_3d_uncertainty(
  class_labels:list, detection_boxes_3d:np.array,detection_scores:list,
  overlapped_thres:float=0.5,overlapped_fn:Callable=overlapped_boxes_3d, 
  appr_factor:float=10.0,top_k:int=-1, attributes:list=None):
  """ Applies non-maximum selection to bounding boxes with uncertainty 
  
  @param class_labels:        list of classes
  @param detection_boxes_3d:  list of bounding boxes
  @param detection_scores:    list of scores
  @param overlapped_thres:    IoU threshold
  @param overlapped_fn:       function to compute overlap
  @param appr_factor:         approximation factor
  @param top_k:               number of bounding boxes to keep
  @param attributes:          list of attributes

  @return: filtered list of classes, scores, bounding boxes, and attributes
  """

  class_labels, detection_scores, detection_boxes_3d, attributes = bboxes_sort( # sort boxes by score
    class_labels, detection_scores, detection_boxes_3d, top_k=top_k,
    attributes=attributes)
  # non maximum supression
  class_labels, detection_scores, detection_boxes_3d, attributes = \
    bboxes_nms_uncertainty(class_labels, detection_scores, detection_boxes_3d,  # apply NMS
                           nms_threshold=overlapped_thres, 
                           overlapped_fn=overlapped_fn,appr_factor=appr_factor, 
                           attributes=attributes)
  
  return class_labels, detection_boxes_3d, detection_scores, attributes


def nms_boxes_3d_merge_only(
  class_labels:list, detection_boxes_3d:np.array,detection_scores:list,
  overlapped_thres:float=0.5,overlapped_fn:Callable=overlapped_boxes_3d, 
  appr_factor:float=10.0,top_k:int=-1, attributes:list=None):
  """ Applies non-maximum selection to bounding boxes by merging them
  
  @param class_labels:        list of classes
  @param detection_boxes_3d:  list of bounding boxes
  @param detection_scores:    list of scores
  @param overlapped_thres:    IoU threshold
  @param overlapped_fn:       function to compute overlap
  @param appr_factor:         approximation factor
  @param top_k:               number of bounding boxes to keep
  @param attributes:          list of attributes

  @return: filtered list of classes, scores, bounding boxes, and attributes
  """
  class_labels, detection_scores, detection_boxes_3d, attributes = bboxes_sort( # sort boxes by score
    class_labels, detection_scores, detection_boxes_3d, top_k=top_k,
    attributes=attributes)
  # nms
  class_labels, detection_scores, detection_boxes_3d, attributes = \
    bboxes_nms_merge_only(                                                      # apply non maximum suppresion 
      class_labels, detection_scores, detection_boxes_3d,
      nms_threshold=overlapped_thres, overlapped_fn=overlapped_fn,
      appr_factor=appr_factor, attributes=attributes)
  
  return class_labels, detection_boxes_3d, detection_scores, attributes


def nms_boxes_3d_score_only(
  class_labels:list, detection_boxes_3d:np.array,detection_scores:list,
  overlapped_thres:float=0.5,overlapped_fn:Callable=overlapped_boxes_3d, 
  appr_factor:float=10.0,top_k:int=-1, attributes:list=None):
  """ Applies non-maximum selection to bounding boxes by selecting highest score 
  
  @param class_labels:        list of classes
  @param detection_boxes_3d:  list of bounding boxes
  @param detection_scores:    list of scores
  @param overlapped_thres:    IoU threshold
  @param overlapped_fn:       function to compute overlap
  @param appr_factor:         approximation factor
  @param top_k:               number of bounding boxes to keep
  @param attributes:          list of attributes

  @return: filtered list of classes, scores, bounding boxes, and attributes
  """
  class_labels, detection_scores, detection_boxes_3d, attributes = bboxes_sort( # sort boxes by score
    class_labels, detection_scores, detection_boxes_3d, top_k=top_k,
    attributes=attributes)
  # nms
  class_labels, detection_scores, detection_boxes_3d, attributes = \
    bboxes_nms_score_only(                                                      # apply non maximum suppresion
      class_labels, detection_scores, detection_boxes_3d,
      nms_threshold=overlapped_thres, overlapped_fn=overlapped_fn,
      appr_factor=appr_factor, attributes=attributes)
  
  return class_labels, detection_boxes_3d, detection_scores, attributes