print('(gyrovector_parallel_transport)')
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

def show(x,y):
  sns.set_style("white")

  xv1 = (torch.tensor((np.sin(np.pi / 3), np.cos(np.pi / 3))) / 5)
  xv2 = (torch.tensor((np.sin(-np.pi / 3), np.cos(np.pi / 3))) / 5)
  t   = (torch.linspace(0, 1, 10)[:, None])

  xy    = pmath.logmap(x, y)
  path  = pmath.geodesic(t, x, y)
  yv1   = pmath.parallel_transport(x, y, xv1)
  yv2   = pmath.parallel_transport(x, y, xv2)

  xgv1 = pmath.geodesic_unit(t, x, xv1)
  xgv2 = pmath.geodesic_unit(t, x, xv2)

  ygv1 = pmath.geodesic_unit(t, y, yv1)
  ygv2 = pmath.geodesic_unit(t, y, yv2)

  R  = 1
  xx = np.linspace(-R, R, N_GRID_EVALS)
  yy = np.sqrt(1-np.pow(xx,2))

  def plot_gv(gv, **kwargs):
    plt.plot(*gv.t().numpy(), **kwargs)
    plt.arrow(*gv[-2], *(gv[-1] - gv[-2]), width=VEC_WIDTH, **kwargs)

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
  plt.annotate(r"$\vec{v}$", x + torch.tensor([0.3, 0.5]), fontsize=FONT_SIZE)
  plot_gv(xgv1, color=COLORS.VECTOR1)
  plot_gv(xgv2, color=COLORS.VECTOR2)
  plt.arrow(*x, *xy, width=0.01, color=COLORS.LINE)
  plot_gv(ygv1, color=COLORS.VECTOR1)
  plot_gv(ygv2, color=COLORS.VECTOR2)

  plt.plot(*path.t().numpy(), color=COLORS.LINE)
  plt.title(r"gyrovector parallel transport $P_{x\to y}$")

  out_file = os.path.join(module_dir, 'out', f'gyro_parallel_transport.png')
  plt.savefig(out_file)
  plt.close()
