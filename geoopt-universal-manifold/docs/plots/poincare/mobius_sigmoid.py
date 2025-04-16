print('(mobius_sigmoid_apply)')
import os
import geoopt.manifolds.poincare.math as pmath
from matplotlib import rcParams
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from globals import COLORS,N_GRID_EVALS,VEC_WIDTH,FONT_SIZE

rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
rcParams["text.usetex"] = True
module_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(module_dir+r'\out', exist_ok=True)

def show(x):
  sns.set_style("white")

  R  = 1
  o = torch.tensor([0,0])
  xx = np.linspace(-R, R, N_GRID_EVALS)
  yy = np.sqrt(1-np.pow(xx,2))
  f_x = pmath.mobius_fn_apply(torch.sigmoid, x)

  plt.fill_between(xx,yy,-yy,color=COLORS.DOMAIN)
  circle = plt.Circle((0, 0), 1, fill=False, color=COLORS.BOUNDARY)
  plt.gca().add_artist(circle)
  lo = -1.1*R
  hi = -lo
  plt.xlim(lo, hi)
  plt.ylim(lo, hi)
  plt.gca().set_aspect("equal")
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])
  plt.annotate("x", x - 0.09, fontsize=FONT_SIZE)
  plt.annotate(r"$\sigma^\otimes(x)$", f_x - torch.tensor([0.1, 0.15]), fontsize=FONT_SIZE)
  plt.arrow(*o, *x, width=VEC_WIDTH, color=COLORS.VECTOR1)
  plt.arrow(*o, *f_x, width=VEC_WIDTH, color=COLORS.VECTOROP)
  plt.title(r"Mobius sigmoid function $\sigma^\otimes(x)$")
  
  out_file = os.path.join(module_dir, 'out', f'mobius_sigmoid.png')
  plt.savefig(out_file)
  plt.close()
