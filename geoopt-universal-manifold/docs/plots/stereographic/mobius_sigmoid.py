print('(möbius_sigmoid)')
import os
from geoopt.manifolds.stereographic import StereographicExact
from matplotlib import rcParams
import torch
import seaborn as sns
from geoopt.manifolds.stereographic.utils import (
  setup_plot, get_interpolation_Ks, get_img_from_fig,
  save_img_sequence_as_boomerang_gif, add_K_box
)
from tqdm import tqdm
from globals import COLORS,N_GRID_EVALS,VEC_WIDTH,FONT_SIZE,GYRO_EVALS
module_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(module_dir+r'\out', exist_ok=True)

rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
rcParams["text.usetex"] = True
sns.set_style("white")


def show(x):
  sns.set_style("white")
  imgs = []

  for K in tqdm(get_interpolation_Ks()):
    sproj_of_sphere = StereographicExact(
      K=K,float_precision=torch.float64,keep_sign_fixed=False,min_abs_K=0.001)
    R = sproj_of_sphere.get_R()
    fig, plt, (lo, hi) = setup_plot(sproj_of_sphere, lo=-2*R-0.1,with_background=True)    

    o     = torch.tensor([0, 0],dtype=torch.float64)
    f_x   = sproj_of_sphere.mobius_fn_apply(torch.sigmoid, x)
    t     = torch.linspace(0, 1, GYRO_EVALS)[:, None]
    xgv   = sproj_of_sphere.geodesic_unit(t, o, x)
    fxgv  = sproj_of_sphere.geodesic_unit(t, o, f_x)


    def plot_gv(gv, **kwargs):
      plt.plot(*gv.t().numpy(), **kwargs)
      plt.arrow(*gv[-2], *(gv[-1] - gv[-2]), width=0.02, **kwargs)

    lo = -2
    hi = -lo

    plot_gv(xgv, color=COLORS.VECTOR1)
    plot_gv(fxgv, color=COLORS.VECTOR1)

    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.gca().set_aspect("equal")
    plt.annotate("x", x - 0.09, fontsize=15)
    plt.annotate(r"$\sigma(x)=\frac{1}{1+e^{-x}}$", x + torch.tensor([-0.7, 0.5]),
                fontsize=FONT_SIZE,color=COLORS.TEXT_COLOR)
    plt.annotate(r"$\sigma^\otimes(x)$", f_x - torch.tensor([0.1, 0.15]),
                fontsize=FONT_SIZE,color=COLORS.TEXT_COLOR)
    # plt.arrow(0, 0, *x, width=0.01, color=COLORS.VECTOR1)
    # plt.arrow(0, 0, *f_x, width=0.01, color=COLORS.VECTOROP)
    plt.title(r" $x \rightarrow \sigma^\otimes(x)$")

    add_K_box(plt, K)
    plt.tight_layout()

    file_name = f"mobius_sigmoid"

    tmp_file =  os.path.join(module_dir, 'tmp', f'{file_name}.png')
    img = get_img_from_fig(fig, tmp_file)
    imgs.append(img)

    # close plot to avoid warnings
    plt.close()

  out_file =  os.path.join(module_dir, 'out', f'{file_name}.gif')
  save_img_sequence_as_boomerang_gif(imgs, out_file)

    