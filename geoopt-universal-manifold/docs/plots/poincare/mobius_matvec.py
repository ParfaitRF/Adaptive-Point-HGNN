print('(Mobius Matvec)')
import os
import geoopt.manifolds.poincare.math as pmath
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from globals import COLORS,N_GRID_EVALS,VEC_WIDTH,FONT_SIZE

rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
rcParams["text.usetex"] = True
module_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(module_dir+r'\out', exist_ok=True)


def show(M,x):
  sns.set_style("white")
  # M   = M.to(device)
  # x   = x.to(device)
  
  R  = 1
  o = torch.tensor([0,0])
  xx = np.linspace(-R, R, N_GRID_EVALS)
  yy = np.sqrt(1-np.pow(xx,2))
  M_x = pmath.mobius_matvec(M, x)

  plt.fill_between(xx,yy,-yy,color=COLORS.DOMAIN)
  circle = plt.Circle((0, 0), R, fill=False, color=COLORS.BOUNDARY)
  plt.gca().add_artist(circle)
  lo = -1.1*R
  hi = -lo
  plt.xlim(lo, hi)
  plt.ylim(lo, hi)
  plt.gca().set_aspect("equal")
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])
  plt.annotate(r"$x$", x - 0.09, fontsize=FONT_SIZE)
  plt.annotate(
    r"$M=\begin{bmatrix}-1 &-1.5\\.2 &.5\end{bmatrix}$",
    x + torch.tensor([-0.5, 0.5]),
    fontsize=FONT_SIZE,
  )
  plt.annotate(r"$M^\otimes x$", M_x - torch.tensor([0.1, 0.15]), fontsize=FONT_SIZE)
  plt.arrow(*o, *x, width=VEC_WIDTH, color=COLORS.VECTOR1)
  plt.arrow(*o, *M_x, width=VEC_WIDTH, color=COLORS.VECTOROP)
  plt.title(r"Matrix multiplication $M\otimes x$")
  
  out_file = os.path.join(module_dir, 'out', f'mobius_matvec.png')
  plt.savefig(out_file)
  plt.close()
