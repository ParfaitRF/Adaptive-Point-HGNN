print('(mobius_matvec)')
import os
from geoopt.manifolds.stereographic import StereographicExact
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from globals import COLORS,N_GRID_EVALS,VEC_WIDTH,FONT_SIZE

module_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(module_dir+r'\out', exist_ok=True)

rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
rcParams["text.usetex"] = True
sns.set_style("white")

module_dir = os.path.dirname(os.path.abspath(__file__))

def show(M,x):
  sproj_of_sphere = StereographicExact(
    K=1.0,float_precision=torch.float64,keep_sign_fixed=False,min_abs_K=0.001)

  # x   = torch.tensor((-0.25, -0.75)) / 3
  # M   = torch.tensor([[-1, -1.5], [0.2, 0.5]])
  M_x = sproj_of_sphere.mobius_matvec(M, x)
  R = 1.0
  xx = np.linspace(-R, R, N_GRID_EVALS)
  yy = np.sqrt(1-np.pow(xx,2))

  
  plt.fill_between(xx,yy,-yy,color=COLORS.DOMAIN)
  circle = plt.Circle((0, 0), 1, fill=False, color=COLORS.BOUNDARY)
  plt.gca().add_artist(circle)
  lo = -2*R-0.1
  hi = -lo
  plt.xlim(lo, hi)
  plt.ylim(lo, hi)
  plt.gca().set_aspect("equal")
  plt.annotate("x", x - 0.09, fontsize=FONT_SIZE)
  plt.annotate(
    r"$M=\begin{bmatrix}-1 &-1.5\\.2 &.5\end{bmatrix}$",
    x + torch.tensor([-0.5, 0.5]),
    fontsize=FONT_SIZE,
  )
  plt.annotate(r"$M^\otimes x$", M_x - torch.tensor([0.1, 0.15]), fontsize=FONT_SIZE)
  plt.arrow(0, 0, *x, width=0.01, color=COLORS.VECTOR1)
  plt.arrow(0, 0, *M_x, width=0.01, color=COLORS.VECTOROP)
  plt.title(r"Matrix multiplication $M\otimes x$")
  plt.show()