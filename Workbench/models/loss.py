from globals import GPU
import torch
from torch import nn#n.nn import Sigmoid,Softmax
import torch.nn.functional as F

from globals import dexp

sigmoid = nn.Sigmoid()


def focal_loss_sigmoid(labels, logits, alpha=0.5, gamma=2):
  """ 
  github.com/tensorflow/models/blob/master/research/object_detection/core/losses.py
  Computer focal loss for binary classification

  @parm labels[list]:   tensor of shape [batch_size].
  @parm logits[list]:   tensor of shape [batch_size].
  @param alpha[float]:  focal loss alpha hyper-parameter.
  @param gamma[int]:    focal loss gamma hyper-parameter.
  """

  prob      = sigmoid(logits)
  labels    = F.one_hot(labels, num_classes=prob.shape[-1]).to(torch.float)

  criterion = nn.BCEWithLogitsLoss(reduce='none')
  
  
  cross_ent     = criterion(logits, labels)
  prob_t        = labels*prob + (1-labels)*(1-prob)
  modulating    = torch.pow(1-prob_t, gamma)
  alpha_weight  = (labels*alpha) + (1-labels)*(1-alpha)
  focal_cross_entropy = modulating * alpha_weight * cross_ent

  return focal_cross_entropy


def focal_loss_softmax(labels, logits, gamma=2):
  """
  https://github.com/fudannlp16/focal-loss/blob/master/focal_loss.py
  Computer focal loss for multi classification

  @params labels[list]: A int32 tensor of shape [batch_size].
  @params logits[list]: A float32 tensor of shape [batch_size,num_classes].
  @params gamma[int]:   A scalar for focal loss gamma hyper-parameter.

  @returns: A tensor of the same shape as `lables`
  """

  y_pred        = F.softmax(logits, dim=-1)                         # get pred. probabilities
  cross_ent     = F.cross_entropy(logits, labels, reduction='none') # cross entropy 
  prob_class    = torch.gather(y_pred, 1, labels.unsqueeze(1))      # predicted probabilities
  focal_factor  = (1 - prob_class) ** gamma                         # modulating factor
  focal_loss    = focal_factor.squeeze(1) * cross_ent               # Final focal loss computation

  return focal_loss


def test_focal_loss():
  n       = 3
  labels  = torch.empty(2, dtype=torch.long).random_(n)
  logits  = torch.tensor([[-100, -100, -100], [-20, -20, -40.0]],
  dtype=torch.float32)#.to(GPU)
  focal_sigmoid = focal_loss_sigmoid(labels,logits)
  focal_softmax = focal_loss_softmax(labels,logits)

  print(focal_sigmoid)
  print(focal_softmax)

if __name__ == '__main__':
  test_focal_loss()

