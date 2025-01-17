""" This file contains the Non-Maximum Suppression algorithm functionalities """

import time
import cv2
import numpy as np
from shapely.geometry import Polygon
from globals import R

SCORES_THRESHOLD  = 0.25
NMS_THRESHOLD     = 0.45
APPR_FACTOR       = 10.0
TOP_K             = 400

def boxes_3d_to_corners(boxes_3d):
  """ returns the 3D corners of bounding boxes

  @param boxes_3d[list]:  list of 3D bounding boxes
  @return [np.array]:     3D corners of bounding boxes  
  """
  
  all_corners = []

  for box_3d in boxes_3d:
    x3d, y3d, z3d, l, h, w, yaw = box_3d
    M_rot   = R(yaw)
    corners = np.array([
      [ l,  0.0,  w],     # front up right
      [ l,  0.0, -w],   # front up left
      [-l,  0.0, -w],  # back up left
      [-l,  0.0,  w],   # back up right   
      [ l, -2*h,  w],   # front down right
      [ l, -2*h, -w],  # front down left
      [-l, -2*h, -w], # back down left
      [-l, -2*h, w]]  # back down right
    )/2
    r_corners = corners.dot(np.transpose(M_rot))
    cam_points_xyz = r_corners + np.array([x3d,y3d,z3d])
    all_corners.append(cam_points_xyz)

  return np.array(all_corners)


def overlapped_boxes_3d(single_box,box_list):
  """ Description
  
  @param single_box[np.array]: single 3D bounding box
  @param box_list[np.array]:   list of 3D bounding boxes

  @return [np.array]:          boxes out of list that overlap with single box
  """

  x0_max,y0_max,z0_max = np.max(single_box,axis=0)
  x0_min,y0_min,z0_min = np.min(single_box,axis=0)
  overlap = np.zeros(len(box_list))

  for i,box in enumerate(box_list):
    xi_max, yi_max, z_max = np.max(box,axis=0)
    x_min, y_min, z_min = np.min(box,axis=0)

    if (x0_max < x_min or x0_min > xi_max or # no overlap in x-axis
        y0_max < y_min or y0_min > yi_max or # no overlap in y-axis
        z0_max < z_min or z0_min > z_max):  # no overlap in z-axis
      overlap[i] = 0
      continue
    
    x_draw_max  = max(x0_max,xi_max)         # max x value
    x_draw_min  = min(x0_min,x_min)         # min x value 
    z_draw_max  = max(z0_max,z_max)         # max z value
    z_draw_min  = min(z0_min,z_min)         # min z value  
    offset      = np.array([x_draw_min, z_draw_min])              # boxes offset from origin
    buf1        = np.zeros((z_draw_max-z_draw_min,x_draw_max-x_draw_min),dtype = np.int32) # TODO: How can we use float values in this function
    buf2        = np.zeros_like(buf1)
    cv2.fillPoly(buf1, [single_box[:4,[0,2]] - offset], color=1)  # TODO: definition missing
    cv2.fillPoly(buf2, [box[:4, [0,2]] - offset], color=1)        # TODO definition missing
    shared_area = cv2.countNonZero(buf1*buf2)                     # TODO: definition missing  (seems to represent area overlap)
    area1       = cv2.countNonZero(buf1)	                        # TODO: definition missing 
    area2       = cv2.countNonZero(buf2)                          # TODO: definition missing 
    shared_y    = min(y0_max,yi_max) - max(y0_min,y_min)           # compute y overlap
    intersection  = shared_y * shared_area                         # compute volume overlap
    union         = (yi_max - y_min) * area2 + (y0_max - y0_min) * area1 #  compute volume union
    overlap[i]    = np.float32(intersection) / (union - intersection)   # IoU

  return overlap


def overlapped_boxes_3d_fast(single_box,box_list):
  """ Description                                     # TODO: detailed description missing
  """ 

  # get max and min valeus of single box in all dimensions
  max_corner = np.max(single_box,axis=0)
  min_corner = np.min(single_box,axis=0)
  x_max,y_max,z_max = max_corner
  x_min,y_min,z_min = min_corner
  max_corners  = np.max(box_list, axis=1)     # max values of all boxes
  min_corners  = np.min(box_list, axis=1)     # min values of all boxes
  overlap      = np.zeros(len(box_list))      # initialize overlap array
  non_overlap_mask = np.logical_or(max_corner < min_corners, min_corner > max_corners) # check for non-overlapping boxes
  non_overlap_mask = np.any(non_overlap_mask, axis=1) # check for non-overlapping boxes
  p1    = Polygon(single_box[:4,[0,2]])       # polygon from single box
  area1 = p1.area                             # single box polygon area 

  for i,box in enumerate(box_list):           # TODO: Why is it we do not utilize the x and z coords. at all  
    if not non_overlap_mask[i]:
      xi_max, yi_max, zi_max = max_corner[i]  # max values of box i
      xi_min, yi_min, zi_min = min_corner[i]  # min values of box i
      p2 = Polygon(box[:4,[0,2]])             # polygon from box i
      xz_overlap  = p1.intersection(p2).area              # xy intersection
      area2         = p2.area                             # list box area
      y_overlap   = min(yi_max,y_max) - max(yi_min,y_min) # y overlap
      overlap  = y_overlap * xz_overlap       # overlapping volume             
      union         = (yi_max - yi_min) * area2 + (y_max-y_min) * area1 - overlap # overlapping volume
      overlap[i]    = np.float32(overlap) / union # IoU

  return overlap


def bboxes_sort(classes,scores,bboxes,top_k=TOP_K,attributes=None):
  """ Sorts bounding boxes by decreasing score order and keep only the top k

  @param classes[np.array]:    classes of bounding boxes
  @param scores[np.array]:     scores of bounding boxes
  @param bboxes[np.array]:     bounding boxes
  @param top_k[int]:           number of top bounding boxes to keep
  @param attributes[np.array]: attributes of bounding boxes

  @return [np.array]:          sorted and truncated indices of bounding boxes
  """

  # sorted indices and rearrange all arrays
  idxs    = np.argsort(-scores,stable=True)

  if top_k > 0: # get only top k indices
    if len(idxs) > top_k:
      idxs = idxs[:top_k]

  classes = classes[idxs]
  scores  = scores[idxs]
  bboxes  = bboxes[idxs]

  if attributes is not None:
    attributes = attributes[idxs]


def bboxes_nms(
  classes,scores,bboxes,nms_threshold=NMS_THRESHOLD,
  overlapped_fn=overlapped_boxes_3d,appr_factor=APPR_FACTOR,attributes=None
):
  """ Apply Non-Maximum Suppression on bounding boxes
  
  @param classes[np.array]:       classes of bounding boxes
  @param scores[np.array]:        scores of bounding boxes
  @param bboxes[np.array]:        bounding boxes
  @param nms_threshold[float]:    threshold for NMS
  @param overlapped_fn[function]: function to compute overlap between boxes
  @param appr_factor[float]:      approximation factor

  @return list:                   classes,scores,bboxes,attributes to be kept
  """

  boxes_corners = boxes_3d_to_corners(bboxes)         # bounding box corners
  boxes_corners = np.int32(boxes_corners*appr_factor) # convert to pixels
  keep_bboxes   = np.ones(scores.shape,dtype=np.bool) # indicates which to keep

  for i in range(scores.size-1):
    if keep_bboxes[i]:
      # compute overlap with all i+j boxes, given i+j <= scores.size-1
      overlap = overlapped_fn(boxes_corners[i], boxes_corners[i+1:])
      # mark boxes that are to be kept
      keep_overlap = np.logical_or(overlap <= nms_threshold,classes[i+1:] != classes[i])
      keep_bboxes[i+1:] = np.logical_and(keep_bboxes[i+1:],keep_overlap)

  # filter boxes to be kept
  idxs    = np.where(keep_bboxes)
  classes = classes[idxs]
  scores  = scores[idxs]
  bboxes  = bboxes[idxs]

  if attributes is not None:
    attributes = attributes[idxs]

  return classes,scores,bboxes,attributes


def bboxes_nms_uncertainty(
  classes,scores,bboxes,scores_threshold=SCORES_THRESHOLD,nms_threshold=NMS_THRESHOLD,
  overlapped_fn=overlapped_boxes_3d_fast,appr_factor=APPR_FACTOR,attributes=None
):
  """ Apply Non-Maximum Suppression on bounding boxes with uncertainty
  
  @param classes[np.array]:       classes of bounding boxes
  @param scores[np.array]:        scores of bounding boxes
  @param bboxes[np.array]:        bounding boxes
  @param scores_threshold[float]: threshold for scores
  @param nms_threshold[float]:    threshold for NMS
  @param overlapped_fn[function]: function to compute overlap between boxes
  @param appr_factor[float]:      approximation factor
  @param attributes[np.array]:    attributes of bounding boxes

  @return list:                   classes,scores,bboxes,attributes to be kept
  """

  boxes_corners = boxes_3d_to_corners(bboxes)
  keep_bboxes   = np.ones(scores.shape,dtype=bool)
  # TODO: Why do we not convert to pixel in this function, although we did in the above defined one

  for i in range(scores.size-1):
    if keep_bboxes[i]:
      valid   = keep_bboxes[i+1:] # valid boxes
      overlap = overlapped_fn(boxes_corners[i],boxes_corners[i+1:][valid])  # compute overlap with current box
      remove_overlap    = np.logical_and( overlap > nms_threshold, classes[i+1:][valid] == classes[i]) # identify overlapping boxes
      overlapped_bboxes = np.concatenate(                                 # group overlapping boxes                     
        [bboxes[(i+1):][valid][remove_overlap], bboxes[[i]]], axis=0)
      boxes_mean          = np.median(overlapped_bboxes,axis=0)           # get the median of all overlapping bboxes
      bboxes[i][:]        = boxes_mean[:]                                   
      box_corners_mean    = boxes_3d_to_corners(np.expand_dims(boxes_mean,axis=0)) # get median corners
      boxes_mean_overlap  = overlapped_fn(box_corners_mean[0],             # compute overlap with the median box
                                           boxes_corners[i+1:][valid][remove_overlap])
      scores[i] += np.sum(                                                # scores weighted by overlaps                
        scores[i+1:][valid][remove_overlap] * boxes_mean_overlap)         
      keep_bboxes[i+1:][valid] = np.logical_not(remove_overlap)           # keep boxes that are not overlapping

  # filter boxes to be kept
  idxs    = np.where(keep_bboxes)
  classes = classes[idxs]
  scores  = scores[idxs]
  bboxes  = bboxes[idxs]

  if attributes is not None:
    attributes = attributes[idxs]

  return classes,scores,bboxes,attributes


def bboxes_nms_merge(
    classes, scores, bboxes, scores_threshold=SCORES_THRESHOLD,nms_threshold=NMS_THRESHOLD,
    overlapped_fn=overlapped_boxes_3d_fast,appr_factor=APPR_FACTOR,attributes=None):
  """ original non-maximum supression

  @param classes[np.array]:       classes of bounding boxes
  @param scores[np.array]:        scores of bounding boxes
  @param bboxes[np.array]:        bounding boxes
  @param scores_threshold[float]: threshold for scores
  @param nms_threshold[float]:    threshold for NMS
  @param overlapped_fn[function]: function to compute overlap between boxes
  @param appr_factor[float]:      approximation factor
  @param attributes[np.array]:    attributes of bounding boxes

  @return list:                   classes,scores,bboxes,attributes to be kept
  """

  boxes_corners = boxes_3d_to_corners(bboxes)         # retrieve box corners
  keep_bboxes = np.ones(scores.shape,dtype=np.bool)   # track boxes to be kept

  for i,box in enumerate(bboxes):
    if i == scores.size-1: continue # no need to irun on last box
    if keep_bboxes[i]:
      valid   = keep_bboxes[i+1:]  # valid boxes
      overlap = overlapped_fn(boxes_corners[i],boxes_corners[i+1:][valid])  # compute overlap with current box
      remove_overlap    = np.logical_and(overlap > nms_threshold,classes[i+1:][valid] == classes[i])   # identify highly overlapping boxes
      overlapped_bboxes = np.concatenate(                   # group overlapping boxes
        [box[valid][remove_overlap], box], axis=0)
      boxes_mean    = np.median(overlapped_bboxes, axis=0)  # get median of all overlapping boxes
      bboxes[i][:]  = boxes_mean[:]
      keep_bboxes[(i+1):][valid] = np.logical_not(remove_overlap)

  # remove highly overlapping boxes
  idxs    = np.where(keep_bboxes)
  classes = classes[idxs]
  scores  = scores[idxs]
  bboxes  = bboxes[idxs]

  if attributes is not None:
    attributes = attributes[idxs]
  
  return classes, scores, bboxes, attributes


def bboxes_nms_score(
  classes, scores, bboxes, scores_threshold=SCORES_THRESHOLD,nms_threshold=NMS_THRESHOLD,
    overlapped_fn=overlapped_boxes_3d_fast,appr_factor=APPR_FACTOR,attributes=None):
    """ apply non-maximum supression and introduce weighted score

    @param classes[np.array]:       classes of bounding boxes
    @param scores[np.array]:        scores of bounding boxes
    @param bboxes[np.array]:        bounding boxes
    @param scores_threshold[float]: threshold for scores
    @param nms_threshold[float]:    threshold for NMS
    @param overlapped_fn[function]: function to compute overlap between boxes
    @param appr_factor[float]:      approximation factor
    @param attributes[np.array]:    attributes of bounding boxes

    @return list:                   classes,scores,bboxes,attributes to be kept
    """

    boxes_corners = boxes_3d_to_corners(bboxes)
    # convert to pixels
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
      if keep_bboxes[i]:
          valid = keep_bboxes[(i+1):] # valid bboxes
          overlap = overlapped_fn(boxes_corners[i],                 # overlap with current bbox
              boxes_corners[(i+1):][valid])                         
          remove_overlap = np.logical_and(overlap > nms_threshold,  # identify bboxes to be removed
              classes[(i+1):][valid] == classes[i])
          overlaped_bboxes = np.concatenate(                        # group overlapping boxes
              [bboxes[(i+1):][valid][remove_overlap], bboxes[[i]]], axis=0)
          boxes_mean    = bboxes[i][:]                              # TODO: What the fuck is this supposed to be?
          bboxes[i][:]  = boxes_mean[:]
          boxes_corners_mean = boxes_3d_to_corners(                 # compute mean
              np.expand_dims(boxes_mean, axis=0))
          boxes_mean_overlap = overlapped_fn(boxes_corners_mean[0], # compute overlap with mean
              boxes_corners[(i+1):][valid][remove_overlap])
          scores[i] += np.sum(                                      # compute weighted scores of high overlaps
              scores[(i+1):][valid][remove_overlap]*boxes_mean_overlap)
          keep_bboxes[(i+1):][valid] = np.logical_not(remove_overlap)

    idxs    = np.where(keep_bboxes)
    classes = classes[idxs]
    scores  = scores[idxs]
    bboxes  = bboxes[idxs]

    if attributes is not None:
        attributes = attributes[idxs]

    return classes, scores, bboxes, attributes


def nms_boxes_3d(
  class_labels, detection_boxes_3d, detection_scores, overlapped_thres=0.5, 
  overlapped_fn=overlapped_boxes_3d, appr_factor=10.0, top_k=-1, attributes=None
):
  """ apply non-maximum supression

  @param classes[np.array]:       classes of bounding boxes
  @param scores[np.array]:        scores of bounding boxes
  @param bboxes[np.array]:        bounding boxes
  @param scores_threshold[float]: threshold for scores
  @param nms_threshold[float]:    threshold for NMS
  @param overlapped_fn[function]: function to compute overlap between boxes
  @param appr_factor[float]:      approximation factor
  @param attributes[np.array]:    attributes of bounding boxes

  @return list:                   classes,scores,bboxes,attributes to be kept
  """
  # sort them by score
  class_labels, detection_scores, detection_boxes_3d, attributes = bboxes_sort( 
    class_labels, detection_scores, detection_boxes_3d, top_k=top_k,attributes=attributes
  )
  # non_maximum supression
  class_labels, detection_scores, detection_boxes_3d, attributes = bboxes_nms(
    class_labels, detection_scores, detection_boxes_3d, nms_threshold=overlapped_thres, 
    overlapped_fn=overlapped_fn, appr_factor=appr_factor, attributes=attributes
  )

  return class_labels, detection_boxes_3d, detection_scores, attributes


def nms_boxes_3d_uncertainty(
  class_labels, detection_boxes_3d, detection_scores, overlapped_thres=0.5, 
  overlapped_fn=overlapped_boxes_3d, appr_factor=10.0, top_k=-1, attributes=None
):
  """ apply non-maximum supression

  @param classes[np.array]:       classes of bounding boxes
  @param scores[np.array]:        scores of bounding boxes
  @param bboxes[np.array]:        bounding boxes
  @param scores_threshold[float]: threshold for scores
  @param nms_threshold[float]:    threshold for NMS
  @param overlapped_fn[function]: function to compute overlap between boxes
  @param appr_factor[float]:      approximation factor
  @param attributes[np.array]:    attributes of bounding boxes

  @return list:                   classes,scores,bboxes,attributes to be kept
  """

  # sort them by score
  class_labels, detection_scores, detection_boxes_3d, attributes = bboxes_sort(
    class_labels, detection_scores, detection_boxes_3d, top_k=top_k,attributes=attributes
  )
  # non-maximum supression
  class_labels, detection_scores, detection_boxes_3d, attributes = bboxes_nms_uncertainty(
    class_labels, detection_scores, detection_boxes_3d, nms_threshold=overlapped_thres, 
    overlapped_fn=overlapped_fn, appr_factor=appr_factor, attributes=attributes
  )

  return class_labels, detection_boxes_3d, detection_scores, attributes


def nms_boxes_3d_merge(
  class_labels, detection_boxes_3d, detection_scores,overlapped_thres=0.5, 
  overlapped_fn=overlapped_boxes_3d, appr_factor=10.0, top_k=-1, attributes=None
):
  """ apply non-maximum supression

  @param classes[np.array]:       classes of bounding boxes
  @param scores[np.array]:        scores of bounding boxes
  @param bboxes[np.array]:        bounding boxes
  @param scores_threshold[float]: threshold for scores
  @param nms_threshold[float]:    threshold for NMS
  @param overlapped_fn[function]: function to compute overlap between boxes
  @param appr_factor[float]:      approximation factor
  @param attributes[np.array]:    attributes of bounding boxes

  @return list:                   classes,scores,bboxes,attributes to be kept
  """

  # non-maximum supression
  class_labels, detection_scores, detection_boxes_3d, attributes = bboxes_sort(
    class_labels, detection_scores, detection_boxes_3d, top_k=top_k, attributes=attributes
  )
  # non-maximum supression
  class_labels, detection_scores, detection_boxes_3d, attributes = bboxes_nms_merge(
    class_labels, detection_scores, detection_boxes_3d, nms_threshold=overlapped_thres, 
    overlapped_fn=overlapped_fn, appr_factor=appr_factor, attributes=attributes
  )

  return class_labels, detection_boxes_3d, detection_scores, attributes


def nms_boxes_3d_score(
  class_labels, detection_boxes_3d, detection_scores, overlapped_thres=0.5, 
  overlapped_fn=overlapped_boxes_3d, appr_factor=10.0, top_k=-1, attributes=None
):
  """ apply non-maximum supression

  @param classes[np.array]:       classes of bounding boxes
  @param scores[np.array]:        scores of bounding boxes
  @param bboxes[np.array]:        bounding boxes
  @param scores_threshold[float]: threshold for scores
  @param nms_threshold[float]:    threshold for NMS
  @param overlapped_fn[function]: function to compute overlap between boxes
  @param appr_factor[float]:      approximation factor
  @param attributes[np.array]:    attributes of bounding boxes

  @return list:                   classes,scores,bboxes,attributes to be kept
  """
  class_labels, detection_scores, detection_boxes_3d, attributes = bboxes_sort(
    class_labels, detection_scores, detection_boxes_3d, top_k=top_k, attributes=attributes
  )
  # non-maximum supression
  class_labels, detection_scores, detection_boxes_3d, attributes = bboxes_nms_score(
    class_labels, detection_scores, detection_boxes_3d, nms_threshold=overlapped_thres, 
    overlapped_fn=overlapped_fn, appr_factor=appr_factor, attributes=attributes
  )
  return class_labels, detection_boxes_3d, detection_scores, attributes