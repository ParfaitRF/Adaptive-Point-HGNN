print('(distance2plane)')
import os
import geoopt.manifolds.poincare.math as pmath
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from globals import COLORS,N_GRID_EVALS,VEC_WIDTH

module_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(module_dir+r'\out', exist_ok=True)

rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
rcParams["text.usetex"] = True


def show(x:torch.Tensor, v:torch.Tensor):
  sns.set_style("white")
  radius = 1
  coords = np.linspace(-radius, radius, N_GRID_EVALS)

  xx, yy  = np.meshgrid(coords, coords)
  dist2   = xx ** 2 + yy ** 2
  mask    = dist2 <= radius ** 2
  grid    = np.stack([xx, yy], axis=-1)
  dists   = pmath.dist2plane(torch.from_numpy(grid).float(), x, v)
  dists[~mask] = np.nan
  circle  = plt.Circle((0, 0), radius, fill=False, color=COLORS.BOUNDARY)
  plt.gca().add_artist(circle)
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])
  plt.xlim(-1.1, 1.1)
  plt.ylim(-1.1, 1.1)

  plt.gca().set_aspect("equal")
  plt.contourf(
    grid[..., 0], grid[..., 1], dists.numpy(), levels=COLORS.RESOLUTION, 
    cmap=COLORS.GRADIENT
  )
  # plt.colorbar()
  plt.scatter(*x, color=COLORS.VECTOR1)
  plt.arrow(*x, *v, color=COLORS.VECTOR1, width=VEC_WIDTH)
  plt.title(r"log distance to $\tilde{H}_{x, v}$")
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])

  out_file = os.path.join(module_dir, 'out', f'distance2plane.png')
  plt.savefig(out_file)
  plt.close()
