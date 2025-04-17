print('(geodesic_grid)')
import os
from geoopt.manifolds.stereographic import StereographicExact
import torch
from geoopt.manifolds.stereographic.utils import setup_plot

module_dir = os.path.dirname(os.path.abspath(__file__))


def show():
  for K in [-1.0,0, 1.0]:
    manifold = StereographicExact(
      K=K,float_precision=torch.float64,keep_sign_fixed=False,min_abs_K=0.001)

    fig, plt, (lo, hi) = setup_plot(manifold, lo=-2.0)

    plt.title(f"Equidistanst grid for  ($\kappa={K:1.0f}$)")

    # use tight layout
    plt.tight_layout()
    out_file = os.path.join(module_dir, 'out', f'geodesic_grid_{K:1.0f}.png')
    plt.savefig(out_file)
