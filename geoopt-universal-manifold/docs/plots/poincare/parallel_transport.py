print('parallel_transport')

import geoopt.manifolds.poincare.math as pmath
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from globals import HEATMAP,COLORS

rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
rcParams["text.usetex"] = True
sns.set_style("white")


def show(x,y,v1,v2,device):
  n_grid_evals = 1000

  x     = x.to(device)
  y     = y.to(device)
  v1    = v1.to(device)
  v2    = v2.to(device)
  t     = torch.linspace(0, 1,n_grid_evals)
  xy    = pmath.logmap(x, y)
  path  = pmath.geodesic(t[:, None], x, y)
  yv1   = pmath.parallel_transport(x, y, v1)
  yv2   = pmath.parallel_transport(x, y, v2)


  circle = plt.Circle((0, 0), 1, fill=False, color=COLORS.blue)
  plt.gca().add_artist(circle)
  plt.xlim(-1.1, 1.1)
  plt.ylim(-1.1, 1.1)
  plt.gca().set_aspect("equal")
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])
  plt.annotate("x", x - 0.07, fontsize=15)
  plt.annotate("y", y - 0.07, fontsize=15)
  plt.annotate(r"$\vec{v}$", x + torch.tensor([0.3, 0.5]), fontsize=15)
  plt.arrow(*x, *v1, width=0.01, color=COLORS.orange)
  plt.arrow(*x, *xy, width=0.01, color=COLORS.blue)
  plt.arrow(*x, *v2, width=0.01, color=COLORS.petrol)
  plt.arrow(*y, *yv1, width=0.01, color=COLORS.orange)
  plt.arrow(*y, *yv2, width=0.01, color=COLORS.petrol)
  plt.plot(*path.t().numpy(), color=COLORS.blue)
  plt.title(r"parallel transport $P^c_{x\to y}$")
  plt.show()
