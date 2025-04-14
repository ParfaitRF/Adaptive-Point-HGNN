print('(mobius_add)')

import geoopt.manifolds.poincare.math as pmath
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from globals import HEATMAP,COLORS

rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
rcParams["text.usetex"] = True


def show(x,y,device):
  sns.set_style("white")

  x = x.to(device)
  y = y.to(device)
  x_plus_y = pmath.mobius_add(x, y)

  circle = plt.Circle((0, 0), 1, fill=False, color=COLORS.blue)
  plt.gca().add_artist(circle)
  plt.xlim(-1.1, 1.1)
  plt.ylim(-1.1, 1.1)
  plt.gca().set_aspect("equal")
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])
  plt.annotate("x", x - 0.09, fontsize=15)
  plt.annotate("y", y - 0.09, fontsize=15)
  plt.annotate(r"$x\oplus y$", x_plus_y - torch.tensor([0.1, 0.15]), fontsize=15)
  plt.arrow(0, 0, *x, width=0.01, color=COLORS.orange)
  plt.arrow(0, 0, *y, width=0.01, color=COLORS.petrol)
  plt.arrow(0, 0, *x_plus_y, width=0.01, color=COLORS.blue)
  plt.title(r"Addition $x\oplus y$")
  plt.show()
