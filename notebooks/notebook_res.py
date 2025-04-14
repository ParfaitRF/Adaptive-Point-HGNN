import os
import sys
import random
import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
from IPython.display import display
from collections import namedtuple

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from data.kitti_dataset import KittiDataset
Hyperplane = namedtuple('Hyperplane',['point','normal']) 
golden_ratio = (1 + np.sqrt(5)) / 2
#from data.crop_aug import save_cropped_boxes,load_cropped_boxes,vis_cropped_boxes

#==============================================================================%
#========================== NON EUCLIDEAN OPERATIONS ==========================%
#==============================================================================%

# hyperbolic trigonometric functions

def tan_k(x:np.ndarray,k:float)->np.ndarray:
  if k < 0:
    return np.tanh(np.sqrt(-k)*x)/np.sqrt(-k)
  if k == 0:
    return x
  else:
    return np.tan(np.sqrt(k)*x)/np.sqrt(k)

def arctan_k(x:np.ndarray,k:float)->np.ndarray:
  if k < 0:
    return np.arctanh(np.sqrt(-k)*x)/np.sqrt(-k)
  if k == 0:
    return x
  else:
    return np.arctan(np.sqrt(k)*x)/np.sqrt(k)
  
def arcsin_k(x:np.ndarray,k:float)->np.ndarray:
  if k < 0:
    return np.arcsinh(np.sqrt(-k)*x)/np.sqrt(-k)
  if k == 0:
    return x
  else:
    return np.arcsin(np.sqrt(k)*x)/np.sqrt(k)
  

#==============================================================================%
#========================== GYROVECTORSPACE OPERATION =========================%
#==============================================================================%

# gyrovector operations
def mobius_add(xs:np.ndarray,ys:np.ndarray,k:float)->np.ndarray:
  axis = 0
  if (xs.shape.__len__() == 1) and (ys.shape.__len__() > 1):
    xs = np.array([xs]*ys.shape[0])
  elif (ys.shape.__len__() == 1) and (xs.shape.__len__() > 1):
    ys = np.array([ys]*xs.shape[0])
  elif (xs.shape.__len__() == 1) and (ys.shape.__len__() == 1):
    xs = xs[np.newaxis,:]
    ys = ys[np.newaxis,:]
  axis = 1

  Z = (1+2*k*np.einsum('nj,nj->n',xs,ys)+k*norm(ys,axis=axis)**2)[:,np.newaxis]*xs+(1-k*norm(xs,axis=axis)**2)[:,np.newaxis]*ys
  N = np.array([1+2*k*np.einsum('nj,nj->n',xs,ys)+(k*norm(xs,axis=axis)*norm(ys,axis=axis))**2])
  N = np.transpose(N)
  return Z/N

def mobius_subtr (y:np.ndarray,x:np.ndarray,k:float)->np.ndarray:
  return mobius_add(-x,y,k)

def mobius_scale(rs:np.ndarray,x:np.ndarray,k:float)->np.ndarray:
  return tan_k(rs*arctan_k(norm(x),k),k)[:,np.newaxis]*x

def distance(ys:np.ndarray,xs:np.ndarray,k:float)->np.ndarray:
  res = []

  for x,y in zip(xs,ys):
    res_subtr = mobius_subtr(y,x,k)
    _dist     = 2*arctan_k(norm(res_subtr),k)
    res.append(_dist)
  
  return np.array(res)

def distance_to_hyperplane(x:np.ndarray,p:np.ndarray,w:np.ndarray,k:float)->float:
  temp        = mobius_subtr(x,p,k)
  numerator   = 2*np.abs(np.dot(temp,w))
  denominator = (1+k*norm(temp)**2)*norm(w)

  return arcsin_k(numerator/denominator,k) if denominator != 0 else ValueError("denominator cannot be 0")

# extended

def radius(k:float)->float:
  return 1/np.sqrt(abs(k)) if k != 0 else ValueError("k cannot be 0")
  
def conformal_factor(x:np.ndarray,k:float)->float:
  return 2 / (1+k*norm(x)**2)

def metric_tensor(x:np.ndarray,k:float)->np.ndarray:
  res =  np.eye(x.shape[0])
  if k != 0:
    res = conformal_factor(x,k)*res
  return res

def inner_product(u,v,x,k):
  lamda = conformal_factor(x,k)**2
  if u.shape.__len__() != 1:
    return lamda*np.einsum('ni,nj->n', u,v)
  else:
    return lamda*np.dot(u,v)

def mobius_norm(v,x,k):
  temp = inner_product(v,v,x,k)
  return np.sqrt(temp)

def angle(u:np.ndarray,v:np.ndarray,k:float)-> float:
  O = np.zeros_like(u)
  norm_u = mobius_norm(u,O,0)
  norm_v = mobius_norm(v,O,0)

  if norm_u == 0 or norm_v == 0:
    raise ValueError("Cannot compute angle with zero vector")
  else:
    return np.arccos(inner_product(u,v,O,0)/(norm_u*norm_v))

def geodesics(x:np.ndarray,y:np.ndarray,k:float,ts:np.array)->np.ndarray:
  dv  = mobius_subtr(y,x,k)
  dvs = mobius_scale(ts,dv,k)
  return mobius_add(x,dvs,k)

def tangent_geodesics(x:np.ndarray,u:np.ndarray,k:float,ts:np.array)->np.ndarray:
  temp = tan_k(ts/2,k)[:,np.newaxis]*u/norm(u)
  return mobius_add(x,temp,k)


# exponential and logarithmic maps

def exponential_map(v:np.ndarray,x:np.ndarray,k:float)->np.ndarray:
  if v.shape.__len__() == 1:  v = np.array([v])
    
  _norm   = conformal_factor(x,k)*norm(v,axis=1)
  _tanx   = tan_k(_norm/2,k)[:,np.newaxis]*v/norm(v,axis=1)[:,np.newaxis]
  _sum    = mobius_add(x,_tanx,k)
  
  return _sum

def log_map(y:np.ndarray,x:np.ndarray,k:float) -> np.ndarray:
  xi = mobius_subtr(y,x,k)
  return 2*arctan_k(norm(xi))*(xi/mobius_norm(xi,x,k))

def antipode(x:np.ndarray,k:float)->np.ndarray:
  if k <= 0:
    return ValueError("k must be positive")
  else:
    conf_fac = conformal_factor(x,k)
    return -x/(conf_fac*k*np.norm(x)**2)
  
def weighted_midpoint(xs:np.ndarray,alphas:np.array,k:float)->np.ndarray:
  conf_factors    = np.array([conformal_factor(x,k) for x in xs])
  scaled_factors  = alphas*conf_factors
  denominator     = np.array([np.sum(scaled_factors-alphas)])

  rs = np.sum(scaled_factors*xs/denominator,axis=0)
  return mobius_scale([0.5],rs,k)[0]