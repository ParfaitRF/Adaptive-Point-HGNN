print('(mobius_add)')
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

def show(x,y,device):
  sns.set_style("white")
  
  o = torch.tensor([0,0])
  x_plus_y = pmath.mobius_add(x, y)

  R  = 1
  xx = np.linspace(-R, R, N_GRID_EVALS)
  yy = np.sqrt(1-np.pow(xx,2))

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
  plt.annotate("x", x - 0.09, fontsize=FONT_SIZE)
  plt.annotate("y", y - 0.09, fontsize=FONT_SIZE)
  plt.annotate(r"$x\oplus y$", x_plus_y - torch.tensor([0.1, 0.15]), fontsize=FONT_SIZE)
  plt.arrow(*o, *x, width=0.01, color=COLORS.VECTOR1)
  plt.arrow(*o, *y, width=0.01, color=COLORS.VECTOR2)
  plt.arrow(*o, *x_plus_y, width=0.01, color=COLORS.VECTOROP)
  plt.title(r"Addition $x\oplus y$")
  
  out_file = os.path.join(module_dir, 'out', f'mobius_add.png')
  plt.savefig(out_file)
  plt.close()
