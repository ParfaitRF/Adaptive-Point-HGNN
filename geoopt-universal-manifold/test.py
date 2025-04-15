import random
import torch
from docs.plots import * 
import numpy as np
from globals import COLORS

gpu = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_global_seed(seed: int):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  # If using CUDA
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def random_pc_vector(dim):
  while True:
    x = (2 * torch.rand(dim)) - 1  # random in (-1, 1)
    if torch.norm(x) < 1:
      return x
    
set_global_seed(42)


x   = random_pc_vector(2)
y   = random_pc_vector(2)
v1  = torch.rand(2)/5
v2  = torch.rand(2)/5
M   = torch.tensor([[-1, -1.5], [0.2, 0.5]])

# POINCARE

#poincare.distance.show(x,device)
#poincare.distance2plane.show(x,v1,device)
#poincare.gyrovector_parallel_transport.show(x,y,device)
#poincare.mobius_add.show(x,y,device)
#poincare.mobius_matvec.show(M,x,device)
#poincare.mobius_sigmoid_apply.show(x,device)
#poincare.parallel_transport.show(x,y,v1,v2,device)

# PRODUCT GEOMETRY

#screenshots = product.torus_embedding.show(device)

# K-STEREOGRAPHIC MODEL

stereographic.distance.show(x,device)