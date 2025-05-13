from sys import float_info
from geoopt.manifolds.base import Manifold
from geoopt import ManifoldParameter
import numpy as np

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F


def instance_normalization(x:torch.Tensor, eps:float=float_info.epsilon): 
  """ Normalizes the input tensor across the batch dimension (dim=0).
  
  Paramters
  ---------
    x: torch.Tensor
      Input tensor of shape (N, C, H, W) or (N, C).
    eps: float
      Small value to avoid division by zero.
  """
  mean = x.mean(dim=0, keepdim=True)
  var = x.var(dim=0, unbiased=False, keepdim=True)
  return (x - mean) / (var + eps).sqrt()


def normalization_fn_riemmannian(
  X:np.ndarray, manifold:Manifold,normalization_fn='IN'
) -> np.ndarray: 
  """ Normalizes the input tensor across the batch dimension (dim=0).
  
  Paramters
  ---------
    x: torch.Tensor
      Input tensor of shape (N, C, H, W) or (N, C).
    eps: float
      Small value to avoid division by zero.
  """

  normalization_fn_dict = {                                                       # Dictionary of normalization functions
    'fused_BN_center':  nn.BatchNorm1d,                                       
    'BN':               partial(nn.BatchNorm1d, affine=False),                   
    'BN_center':        nn.BatchNorm1d,                                           
    'IN':               instance_normalization,
    'NONE':             None
  }

  normalization_fn = normalization_fn_dict[normalization_fn]                     # Get the normalization function
  
  X = ManifoldParameter(X, manifold=manifold)                                   # Convert to ManifoldParameter
  base_point  = torch.zeros_like(X[0])
  x_tangent     = manifold.logmap(X, base_point)                                  # Projects to tangent space
  normalized  = normalization_fn(x_tangent)                                       # Normalize the tangent space                 
  x_back      = manifold.expmap(normalized, base_point)                         # Projects back to the manifold

  return x_back   


activation_fn_dict = {                                                          # Dictionary of activation functions    
  'ReLU':      F.relu,
  'ReLU6':     F.relu6,
  'LeakyReLU': partial(F.leaky_relu, negative_slope=0.01),
  'ELU':       F.elu,
  'Sigmoid':   torch.sigmoid,
  'Tanh':      torch.tanh,
  'NONE':      None
}


def multi_layer_fc_fn(sv, mask=None, Ks=(64, 32, 64), num_classes=4,
  is_logits=False, normalization_type="fused_BN_center",activation_type='ReLU'):
  """
  A function to create multiple layers of neural network to compute features 
  passing through each edge.

  Paramters
  ---------
  sv: [n, m] or [t, degree, M] tensor.
    n is the total number of edges, m is the length of features. t is
    the number of receiving vertices, degree is the in-degree of each
    recieving vertices. When a [t, degree, m] tensor is provided, the
    degree of each recieving vertex is assumed to be same.
  mask: optional [n, 1] or [t, degree, 1] tensor. 
    A value 1 is used to indicate a valid output feature, while a value 0 
    indicates an invalid output feature which is set to 0.
  Ks: Tuple[int]:
    A tuple of integers representing the number of features in each layer.
    The length of the tuple is equal to the number of layers minus 1.
    The last layer will have the same number of features as the number of
    classes.
  num_classes: int:
    The number of classes for the final output layer.
  is_logits: bool:
    If True, the final layer will not have an activation function.
  num_layer: int:
    The number of layers in the neural network.

  Returns
  _______
    a [n, k] tensor or [t, degree, k].
    k is the length of the new features on the edge.
  """
  
  assert len(sv.shape) == 2
  
  if is_logits:
    features = sv
    for i in range(len(Ks)):
      features = slim.fully_connected(
        features, 
        Ks[i],
        activation_fn=activation_fn_dict[activation_type],
        normalizer_fn=normalization_fn_dict[normalization_type],
      )
    features = slim.fully_connected(
      features, 
      num_classes,
      activation_fn=None,
      normalizer_fn=None
    )
  else:
      features = sv
      for i in range(len(Ks)):
        features = slim.fully_connected(
          features, 
          Ks[i],
          activation_fn=activation_fn_dict[activation_type],
          normalizer_fn=normalization_fn_dict[normalization_type],
        )
      features = slim.fully_connected(
        features, 
        num_classes,
        activation_fn=activation_fn_dict[activation_type],
        normalizer_fn=normalization_fn_dict[normalization_type],
      )
  if mask is not None:
      features = features * mask
  return features


class MultiLayerFC(nn.Module):
  """ A class to create multiple layers of neural network to compute features """
  def __init__(
    self, input_dim, Ks=(64, 32, 64), num_classes=4,is_logits=False, 
    activation_type='ReLU', normalization_type='BN'
  ):
    """ Initializes the MultiLayerFC class.
    
    Parameters
    ----------
    input_dim: int
      The input dimension of first layer (feature dimension).
    Ks: Tuple[int]
      A tuple of integers representing the number of features in each layer.
      The length of the tuple is equal to the number of layers minus 1.
      The last layer will have the same number of features as the number of
      classes.
    num_classes: int
      The number of classes for the final output layer.
    is_logits: bool
      If True, the final layer will not have an activation function.
    activation_type: str
      The activation function to be used in the layers. Default is 'ReLU'.
      Other options include 'ReLU6', 'LeakyReLU', 'ELU', 'Sigmoid', 'Tanh'.
    """
    super(MultiLayerFC, self).__init__()
    
    self.is_logits  = is_logits
    self.layers     = nn.ModuleList()
    self.norms      = nn.ModuleList()

    all_dims    = [input_dim] + list(Ks) + [num_classes]
    num_layers  = len(all_dims) - 1

    # Choose activation function
    self.activation_fn = getattr(F, activation_type.lower(), F.relu)

    for i in range(num_layers):
      self.layers.append(nn.Linear(all_dims[i], all_dims[i+1]))
      if normalization_type == "BN":
        self.norms.append(nn.BatchNorm1d(all_dims[i+1]))
      elif normalization_type == "LN":
        self.norms.append(nn.LayerNorm(all_dims[i+1]))
      else:
        self.norms.append(None)  # No normalization

  def forward(self, sv, mask=None):
    x = sv
    for i in range(len(self.layers)):
      x = self.layers[i](x)
      if self.norms[i] is not None:
        x = self.norms[i](x)
      if not self.is_logits or i < len(self.layers) - 1:
        x = self.activation_fn(x)
    
    if mask is not None:
      x = x * mask
    return x