print('(midpoint)')
import os
from geoopt.manifolds.stereographic import StereographicExact
import torch
import numpy as np
from geoopt.manifolds.stereographic.utils import (
  setup_plot, get_interpolation_Ks, get_img_from_fig,
  save_img_sequence_as_boomerang_gif, add_K_box
)
    
from tqdm import tqdm
from globals import COLORS
module_dir = os.path.dirname(os.path.abspath(__file__))

def show(x,y,v1,v2):
  imgs = []

  # for every K of the interpolation sequence
  for K in tqdm(get_interpolation_Ks()):
    # create manifold for K
    manifold = StereographicExact(
      K=K,float_precision=torch.float64,keep_sign_fixed=False,min_abs_K=0.001)

    # set up plot
    fig, plt, (lo, hi) = setup_plot(manifold, lo=-3.0)

    # get manifold properties
    K = manifold.get_K().item()
    R = manifold.get_R().item()

    t     = torch.linspace(0, 1, 200)[:, None]
    xy    = manifold.logmap(x, y)
    path  = manifold.geodesic(t, x, y)
    yv1   = manifold.transp(x, y, v1)
    yv2   = manifold.transp(x, y, v2)

    xgv1 = manifold.geodesic_unit(t, x, v1)
    xgv2 = manifold.geodesic_unit(t, x, v2)

    ygv1 = manifold.geodesic_unit(t, y, yv1)
    ygv2 = manifold.geodesic_unit(t, y, yv2)

    def plot_gv(gv, **kwargs):
      plt.plot(*gv.t().numpy(), **kwargs)
      plt.arrow(*gv[-2], *(gv[-1] - gv[-2]), width=0.02, **kwargs)

    plt.annotate("$x$", x - 0.15, fontsize=15, color=COLORS.TEXT_COLOR)
    plt.annotate("$y$", y + torch.tensor([0.05, -0.15]), fontsize=15, color=COLORS.TEXT_COLOR)
    plt.annotate(r"$\vec{v}$", x + xy+ 0.05, fontsize=15, color=COLORS.TEXT_COLOR)

    plot_gv(xgv1, color=COLORS.MAT_RED)
    plot_gv(xgv2, color=COLORS.SHINY_BLUE)
    plt.arrow(*x, *xy, width=0.01, color=COLORS.SHINY_GREEN)
    plot_gv(ygv1, color=COLORS.MAT_RED)
    plot_gv(ygv2, color=COLORS.SHINY_BLUE)
    plt.plot(*path.t().numpy(), color=COLORS.MAT_YELLOW)

    # add plot title
    plt.title(r"Gyrovector Parallel Transport $P^\kappa_{x\to y}$")

    # add curvature box
    add_K_box(plt, K)

    # use tight layout
    plt.tight_layout()

    # convert plot to image array
    tmp_file =  os.path.join(module_dir, 'tmp', 'gyrovector-parallel-transport.png')
    img = get_img_from_fig(fig, tmp_file)
    imgs.append(img)

    # close plot to avoid warnings
    plt.close()

  # save img sequence as infinite boomerang gif
  out_file =  os.path.join(module_dir, 'out', 'gyrovector-parallel-transport.gif')
  save_img_sequence_as_boomerang_gif(imgs, out_file)
