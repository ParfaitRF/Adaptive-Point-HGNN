print('(mobius_add)')
import os
from geoopt.manifolds.stereographic import StereographicExact
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from geoopt.manifolds.stereographic.utils import (
  setup_plot, get_interpolation_Ks, get_img_from_fig,
  save_img_sequence_as_boomerang_gif, add_K_box
)
import numpy as np
from tqdm import tqdm
from globals import COLORS
module_dir = os.path.dirname(os.path.abspath(__file__))

rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
rcParams["text.usetex"] = True
sns.set_style("white")

def show(x,y):
  imgs = []

  for K in tqdm(get_interpolation_Ks()):

  # # POINCARE BALL MOBIUS ADD PLOT ################################################
  # poincare_ball = StereographicExact(
  #   K=-1.0,float_precision=torch.float64,keep_sign_fixed=False,min_abs_K=0.001
  # )
  # fig, plt, (lo, hi) = setup_plot(poincare_ball, lo=-2.0)
  # x_plus_y = poincare_ball.mobius_add(x, y)
  # circle = plt.Circle((0, 0), 1, fill=False, color="b")
  # plt.gca().add_artist(circle)
  # plt.xlim(-1.1, 1.1)
  # plt.ylim(-1.1, 1.1)
  # plt.gca().set_aspect("equal")
  # plt.annotate("x", x - 0.09, fontsize=15)
  # plt.annotate("y", y - 0.09, fontsize=15)
  # plt.annotate(r"$x\oplus y$", x_plus_y - torch.tensor([0.1, 0.15]), fontsize=15)
  # plt.arrow(0, 0, *x, width=0.01, color="r")
  # plt.arrow(0, 0, *y, width=0.01, color="g")
  # plt.arrow(0, 0, *x_plus_y, width=0.01, color="b")
  # plt.title(r"Addition $x\oplus y$")
  # plt.show()


  # SPROJ OF SPHERE MOBIUS ADD PLOT ##############################################

    sproj_of_sphere = StereographicExact(
      K=K,float_precision=torch.float64,keep_sign_fixed=False,min_abs_K=0.001
    )
    R = sproj_of_sphere.get_R()
    fig, plt, (lo, hi) = setup_plot(sproj_of_sphere, lo=-2*R-0.1)
    x /= 1.5
    y /= 1.5
    x_plus_y = sproj_of_sphere.mobius_add(x, y)

    # get gyrovectors
    o   = torch.tensor([0.0, 0.0],dtype=torch.float64)
    t   = torch.linspace(0, 1, 200)[:, None]
    xg  = sproj_of_sphere.geodesic_unit(t,o, x)
    yg  = sproj_of_sphere.geodesic_unit(t,o, y)
    xyg = sproj_of_sphere.geodesic_unit(t,o, x_plus_y)

    circle = plt.Circle((0, 0), R, fill=False, color=COLORS.GREY)
    plt.gca().add_artist(circle)
    
    def plot_gv(gv, **kwargs):
      plt.plot(*gv.t().numpy(), **kwargs)
      plt.arrow(*gv[-2], *(gv[-1] - gv[-2]), width=0.02, **kwargs)

    plt.annotate("x", x - 0.09, fontsize=15)
    plt.annotate("y", y - 0.09, fontsize=15)
    plt.annotate(r"$x\oplus y$", x_plus_y - torch.tensor([0.1, 0.15]), fontsize=15)
    lo = -R*1.1
    hi = -lo
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plot_gv(xg, color=COLORS.MAT_RED)
    plot_gv(yg, color=COLORS.SHINY_GREEN)
    plot_gv(xyg, color=COLORS.MAT_YELLOW)
    plt.title(r"Addition $x\oplus y$")

    add_K_box(plt, K)
    plt.tight_layout()

    tmp_file =  os.path.join(module_dir, 'tmp', 'mobius_add.png')
    img = get_img_from_fig(fig, tmp_file)
    imgs.append(img)

    # close plot to avoid warnings
    plt.close()

  # save img sequence as infinite boomerang gif
  out_file =  os.path.join(module_dir, 'out', 'mobius_add.gif')
  save_img_sequence_as_boomerang_gif(imgs, out_file)
