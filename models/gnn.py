from functools import partial

import torch
import torch.nn as nn
from globals import GPU




class InstanceNormalization(nn.Module):
  def __init__(self, num_features, eps=1e-12):
    super(InstanceNormalization, self).__init__()
    self.inst_norm = nn.InstanceNorm1d(num_features, eps=eps, affine=False)

  def forward(self, x):
    return self.inst_norm(x)
  

# Normalization functions dictionary
normalization_fn_dict = {
  'BN': nn.BatchNorm2d,         # Batch Normalization for 2D inputs
  'IN': InstanceNormalization,  # Using the custom class we defined
  'NONE': None                  # No normalization
}

# Activation functions dictionary
# https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
activation_fn_dict = {
  'ReLU': nn.ReLU,
  'ReLU6': nn.ReLU6,
  'LeakyReLU': nn.LeakyReLU,
  'ELU': nn.ELU,
  'NONE': None,               # No activation function
  'Sigmoid': nn.Sigmoid,
  'Tanh': nn.Tanh,
}


def multi_layer_fc_fn(sv, mask=None, Ks=(64, 32, 64), num_classes=4,
  is_logits=False, num_layer=4, normalization_type="fused_BN_center",
  activation_type='ReLU'):
  """
  A function to create multiple layers of neural network to compute
  features passing through each edge.

  @param sv tensor:   a [N,M] or [T, DEGREE, M] tensor, where N is the number f edges, 
                      M is the length of features. T is the number of received vertices.
                      When a [T, DEGREE, M] tensor is provided, the degree is assumed 
                      to be vertex invariant.
  @param mask tensor: a optional [N, 1] or [T, DEGREE, 1] tensor. A value 1 is used
                      to indicate a valid output feature, while a value 0 indicates
                      an invalid output feature which is set to 0.
  @param Ks:          secifies the number of neurons in each layer
  @param num_classes: # TODO: number of classes
  @param is_logits:   # TODO: explain
  @param num_layer:   # TODO: number of layers to add
  @param normalization_type: # TODO: explain
  @param activation_type:    # TODO: explain
  """

  assert len(sv.shape) == 2
  assert len(Ks) == num_layer-1

  if is_logits:
    features = sv

  else:
    pass