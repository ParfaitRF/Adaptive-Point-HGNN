import torch
from torchvision.ops import (
  sigmoid_focal_loss,                                                           # https://pytorch.org/vision/master/generated/torchvision.ops.sigmoid_focal_loss.html

)
import torch.nn.functional as F


def focal_loss_softmax(logits, labels, gamma=2.0, reduction='none'):
  """
  Focal loss for multi-class classification.
  
  Parameters
  ----------
    logits: Tensor of shape [batch_size, num_classes]
    labels: Tensor of shape [batch_size] (int64)
    gamma: focusing parameter
    reduction: 'none' | 'mean' | 'sum'
  Returns
  -------
    Loss tensor according to the specified reduction.
  """
  log_probs = F.log_softmax(logits, dim=-1)  # [B, C]
  probs = torch.exp(log_probs)               # [B, C]
  labels = labels.long()
  
  ce_loss = F.nll_loss(log_probs, labels, reduction='none')                     # Cross entropy
  pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)                          # Get probs of true class
  focal_factor = (1 - pt) ** gamma
  
  loss = focal_factor * ce_loss

  if reduction == 'mean':
      return loss.mean()
  elif reduction == 'sum':
      return loss.sum()
  return loss


