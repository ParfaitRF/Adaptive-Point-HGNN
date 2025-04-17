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
from tqdm import tqdm
from globals import COLORS,N_GRID_EVALS,FONT_SIZE,VEC_WIDTH,GYRO_EVALS

module_dir = os.path.dirname(os.path.abspath(__file__))

rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
rcParams["text.usetex"] = True
sns.set_style("white")

def show(x,y):
  imgs = []

  for K in tqdm(get_interpolation_Ks()):

  # SPROJ OF SPHERE MOBIUS ADD PLOT ##############################################

    sproj_of_sphere = StereographicExact(
      K=K,float_precision=torch.float64,keep_sign_fixed=False,min_abs_K=0.001
    )
    R = sproj_of_sphere.get_R()
    fig, plt, (lo, hi) = setup_plot(sproj_of_sphere, lo=-2*R-0.1)
    x_plus_y = sproj_of_sphere.mobius_add(x, y)

    plt.annotate(r"$x$", x-0.15, fontsize=FONT_SIZE,color=COLORS.TEXT_COLOR)
    plt.annotate(r"$y$", y-0.15, fontsize=FONT_SIZE,color=COLORS.TEXT_COLOR)
    plt.annotate(r"$x \oplus y$", x_plus_y-0.15,fontsize=FONT_SIZE,color=COLORS.TEXT_COLOR)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.arrow(0, 0, *x, width=0.01, color=COLORS.VECTOR1)
    plt.arrow(0, 0, *y, width=0.01, color=COLORS.VECTOR2)
    plt.arrow(0, 0, *x_plus_y, width=0.01, color=COLORS.VECTOROP)
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
