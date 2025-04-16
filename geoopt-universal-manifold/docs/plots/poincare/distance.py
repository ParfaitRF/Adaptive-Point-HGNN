print('(distance)')
import os
import geoopt.manifolds.poincare.math as pmath
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from globals import COLORS,N_GRID_EVALS
module_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(module_dir+r'\out', exist_ok=True)

def show(x):
  sns.set_style("white")
  R  = 1
  coords  = np.linspace(-R, R, 100)
  xx, yy  = np.meshgrid(coords, coords)
  dist2   = xx ** 2 + yy ** 2
  mask    = dist2 <= R ** 2
  grid    = np.stack([xx, yy], axis=-1)
  dists   = pmath.dist(torch.from_numpy(grid).float(), x)
  dists[~mask] = np.nan
  circle  = plt.Circle((0, 0), 1, fill=False, color=COLORS.BOUNDARY)
  plt.gca().add_artist(circle)
  lo = -1.1*R
  hi = -lo
  plt.xlim(lo, hi)
  plt.ylim(lo, hi)
  plt.gca().set_aspect("equal")
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])
  plt.contourf(
    grid[..., 0], grid[..., 1], dists, levels=COLORS.RESOLUTION, cmap=COLORS.GRADIENT
  )
  plt.colorbar()
  plt.title(
    r"log distance to $x$".format(round(x[0].item(), 2), round(x[1].item(), 2))
  )
  out_file = os.path.join(module_dir, 'out', f'distance.png')
  plt.savefig(out_file)
  plt.close()
