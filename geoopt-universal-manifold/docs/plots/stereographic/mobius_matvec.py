print('(mobius_matvec)')
import os
from geoopt.manifolds.stereographic import StereographicExact
import torch
import seaborn as sns
from matplotlib import rcParams
from geoopt.manifolds.stereographic.utils import (
  setup_plot, get_interpolation_Ks, get_img_from_fig,
  save_img_sequence_as_boomerang_gif, add_K_box
)
from tqdm import tqdm
from globals import COLORS,N_GRID_EVALS,VEC_WIDTH,FONT_SIZE
module_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(module_dir+r'\out', exist_ok=True)

rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
rcParams["text.usetex"] = True
sns.set_style("white")

module_dir = os.path.dirname(os.path.abspath(__file__))

def show(M,x):
  imgs = []

  for K in tqdm(get_interpolation_Ks()):

    sproj_of_sphere = StereographicExact(
      K=K,float_precision=torch.float64,keep_sign_fixed=False,min_abs_K=0.001)
    R = sproj_of_sphere.get_R()
    fig, plt, (lo, hi) = setup_plot(sproj_of_sphere, lo=-2*R-0.1,with_background=True)
    M_x = sproj_of_sphere.mobius_matvec(M, x)

    lo = -2
    hi = -lo
    
    plt.gca().set_aspect("equal")
    plt.annotate("x", x - 0.09, fontsize=FONT_SIZE,color=COLORS.TEXT_COLOR)
    plt.annotate(
      r"$M=\begin{bmatrix}-1 &-1.5\\.2 &.5\end{bmatrix}$",
      x + torch.tensor([-0.5, 0.5]),
      fontsize=FONT_SIZE,color=COLORS.TEXT_COLOR
    )
    plt.annotate(r"$M^\otimes x$", M_x - torch.tensor([0.1, 0.15]), 
                 fontsize=FONT_SIZE,color=COLORS.TEXT_COLOR)
    plt.arrow(0, 0, *x, width=0.01, color=COLORS.VECTOR1)
    plt.arrow(0, 0, *M_x, width=0.01, color=COLORS.VECTOROP)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.title(r"Matrix multiplication $M\otimes x$")

    add_K_box(plt, K)
    plt.tight_layout()

    file_name = f"mobius_matvec"

    tmp_file =  os.path.join(module_dir, 'tmp', f'{file_name}.png')
    img = get_img_from_fig(fig, tmp_file)
    imgs.append(img)

    # close plot to avoid warnings
    plt.close()
  
  out_file =  os.path.join(module_dir, 'out', f'{file_name}.gif')
  save_img_sequence_as_boomerang_gif(imgs, out_file)