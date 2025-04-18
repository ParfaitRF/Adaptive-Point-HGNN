print('(distance)')
import os
from geoopt.manifolds.stereographic import StereographicExact
import torch
import numpy as np
from geoopt.manifolds.stereographic.utils import (
  setup_plot, get_interpolation_Ks, get_img_from_fig,
  save_img_sequence_as_boomerang_gif, add_K_box
)
from tqdm import tqdm
from globals import COLORS,N_GRID_EVALS,FONT_SIZE

module_dir = os.path.dirname(os.path.abspath(__file__))

def show(x:torch.Tensor):
  n_grid_evals = N_GRID_EVALS
  imgs = []

  # for every K of the interpolation sequence
  max_dist = 0
  for K in tqdm(get_interpolation_Ks()):
    # create manifold for K
    manifold = StereographicExact(
      K=K,float_precision=torch.float64,keep_sign_fixed=False,min_abs_K=0.001
    )

    # set up plot
    fig, plt, (lo, hi) = setup_plot(manifold, lo=-3.0, with_background=False)

    # get manifold properties
    K = manifold.get_K().item()
    R = manifold.get_R().item()

    # create mesh-grid
    coords = None
    if K < 0:
      coords = np.linspace(lo, hi, n_grid_evals)
    else:
      coords = np.linspace(lo, hi, n_grid_evals)
    xx, yy  = np.meshgrid(coords, coords)
    grid    = np.stack([xx, yy], axis=-1)

    # compute distances to point
    dists = manifold.dist(torch.from_numpy(grid).float(), x)

    # zero-out points outside of Poincaré ball
    if K < 0:
      dist2 = xx ** 2 + yy ** 2
      mask  = dist2 <= R ** 2
      dists[(~mask).nonzero()] = np.nan
      dists[mask] = dists[mask].sqrt()
      levels      = np.linspace(0, dists[mask].max(), COLORS.RESOLUTION)
    else:
      dists       = dists.sqrt()
      levels      = np.linspace(0, dists.max(), COLORS.RESOLUTION)

    max_dist = max(max_dist,levels[-1])
    #add contour plot
    plt.contourf(
      grid[..., 0],
      grid[..., 1],
      dists,
      levels=levels,
      cmap=COLORS.GRADIENT
    )
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_ticks(np.arange(0, levels[-1], 0.5))

    # plot x
    plt.scatter(*x, s=3.0, color=COLORS.POINT1)
    plt.annotate("$x$", x + torch.tensor([-0.15, 0.05]), fontsize=FONT_SIZE, color=COLORS.POINT1)

    # add plot title
    plt.title(r"Square Root of Distance to $x$")

    # add curvature box
    add_K_box(plt, K)

    # use tight layout
    plt.tight_layout()

    file_name = 'distance'
    # convert plot to image array
    tmp_file = os.path.join(module_dir, 'tmp', f'{file_name}.png')
    img = get_img_from_fig(fig, tmp_file)
    imgs.append(img)

    # close plot to avoid warnings
    plt.close()
  print(max_dist)
  # save img sequence as infinite boomerang gif
  out_file = os.path.join(module_dir, 'out', f'{file_name}.gif')
  save_img_sequence_as_boomerang_gif(imgs, out_file)
