print('(distance)')

import geoopt.manifolds.poincare.math as pmath
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from globals import HEATMAP,COLORS


def show(x,device):
  sns.set_style("white")
  radius  = 1
  coords  = np.linspace(-radius, radius, 100)
  x       = x.to(device)
  xx, yy  = np.meshgrid(coords, coords)
  dist2   = xx ** 2 + yy ** 2
  mask    = dist2 <= radius ** 2
  grid    = np.stack([xx, yy], axis=-1)
  dists   = pmath.dist(torch.from_numpy(grid).float().to(device), x)
  dists[(~mask).nonzero()] = np.nan
  circle  = plt.Circle((0, 0), 1, fill=False, color=COLORS.BLUE)
  plt.gca().add_artist(circle)
  plt.xlim(-1.1, 1.1)
  plt.ylim(-1.1, 1.1)
  plt.gca().set_aspect("equal")
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])
  plt.contourf(
    grid[..., 0], grid[..., 1], dists, levels=100, cmap=HEATMAP
  )
  plt.colorbar()
  plt.title(
    r"log distance to $x$".format(round(x[0].item(), 2), round(x[1].item(), 2))
  )
  plt.show()
