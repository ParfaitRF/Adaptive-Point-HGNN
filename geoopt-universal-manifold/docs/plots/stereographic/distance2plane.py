print('(distance2plane)')
import os
import shutil
from geoopt.manifolds.stereographic import StereographicExact
import torch
import numpy as np
from geoopt.manifolds.stereographic.utils import (
  setup_plot, get_interpolation_Ks, get_img_from_fig,
  save_img_sequence_as_boomerang_gif, add_K_box
)
from tqdm import tqdm
from globals import COLORS,N_GRID_EVALS,HEATMAP,COLOR_RES

module_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(module_dir+r'\tmp', exist_ok=True)

def show(x:torch.Tensor, v:torch.Tensor):
  n_grid_evals = N_GRID_EVALS
  imgs = []

  # for every K of the interpolation sequence
  for K in tqdm(get_interpolation_Ks()):
    # create manifold for K
    manifold = StereographicExact(
      K=K,float_precision=torch.float64,keep_sign_fixed=False,min_abs_K=0.001
    )

    # set up plot
    fig, plt, (lo, hi) = setup_plot(
      manifold, lo=-3.0, grid_line_width=0.25, with_background=False)

    # get manifold properties
    K = manifold.get_K().item()
    R = manifold.get_R().item()

    # create point on plane x and normal vector v
    # x = torch.tensor([-0.75, 0])
    # v = torch.tensor([0.5, -1 / 3])

    # create grid mesh
    coords = None
    if K < 0:
      coords = np.linspace(lo, hi, n_grid_evals)
    else:
      coords = np.linspace(lo, hi, n_grid_evals)
    xx, yy  = np.meshgrid(coords, coords)
    grid    = np.stack([xx, yy], axis=-1)

    # compute distances to hyperplane
    dists = manifold.dist2plane(torch.from_numpy(grid).float(), x, v)

    # zero-out points outside of Poincaré ball
    if K < 0:
      dist2 = xx ** 2 + yy ** 2
      mask  = dist2 <= R ** 2
      dists[(~mask).nonzero()] = np.nan
      # add contour plot
    
    plt.contourf(
      grid[..., 0],
      grid[..., 1],
      dists.sqrt(),
      levels  = np.linspace(0,5,COLOR_RES),
      cmap    = HEATMAP
    )
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    
    cbar.set_ticks(np.arange(0, 5, 0.5))

    # plot x
    plt.annotate("$p$", x + torch.tensor([-0.15, 0.05]), fontsize=15,color=COLORS.TEXT_COLOR)
    plt.scatter(*x, s=20.0, color=COLORS.TEXT_COLOR)

    # plot vector from x to v
    plt.annotate("$\\vec{w}$", x + v +torch.tensor([-0.0, 0.12]), fontsize=15,
                color=COLORS.TEXT_COLOR)
    #plt.arrow(*x, *v, color=COLORS.TEXT_COLOR, width=0.02)

    # add plot title
    # plt.title(r"Square Root of Distance to $\tilde{H}_{p, w}$")

    # add curvature box
    add_K_box(plt, K)

    # use tight layout
    plt.tight_layout()

    # convert plot to image array
    img = get_img_from_fig(fig, module_dir+r'\tmp\distance2plane.png')
    imgs.append(img)

    # close plot to avoid warnings
    plt.close()
  
  os.remove(module_dir+r'\tmp\distance2plane.png')
  # save img sequence as infinite boomerang gif
  save_img_sequence_as_boomerang_gif(imgs, module_dir+r'\out\distance2plane.gif')
