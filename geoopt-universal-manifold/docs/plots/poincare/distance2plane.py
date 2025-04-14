print('(distance2plane)')

import geoopt.manifolds.poincare.math as pmath
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from globals import HEATMAP,COLORS

rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
rcParams["text.usetex"] = True


def show(x:torch.Tensor, v:torch.Tensor,device):
  sns.set_style("white")
  radius = 1
  coords = np.linspace(-radius, radius, 100)
  x = x.to(device)
  v = v.to(device)

  xx, yy  = np.meshgrid(coords, coords)
  dist2   = xx ** 2 + yy ** 2
  mask    = dist2 <= radius ** 2
  grid    = np.stack([xx, yy], axis=-1)
  dists   = pmath.dist2plane(torch.from_numpy(grid).float().to(device), x, v)
  dists[(~mask).nonzero()] = np.nan
  circle  = plt.Circle((0, 0), 1, fill=False, color="b")
  plt.gca().add_artist(circle)
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])
  plt.xlim(-1.1, 1.1)
  plt.ylim(-1.1, 1.1)

  plt.gca().set_aspect("equal")
  plt.contourf(
    grid[..., 0], grid[..., 1], dists.numpy(), levels=200, cmap=HEATMAP
  )
  plt.colorbar()
  plt.scatter(*x, color=COLORS.PETROL)
  plt.arrow(*x, *v, color=COLORS.PETROL, width=0.01)
  plt.title(r"log distance to $\tilde{H}_{x, v}$")
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])
  plt.show()
