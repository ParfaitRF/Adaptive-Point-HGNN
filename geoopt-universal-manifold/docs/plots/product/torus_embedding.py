print('(torus_embedding)')

import torch
from geoopt import ManifoldTensor, ManifoldParameter
from geoopt.manifolds import SphereExact, Scaled, ProductManifold
from geoopt.optim import RiemannianAdam
import numpy as np
from numpy import pi, cos, sin
from mayavi import mlab
import imageio
from tqdm import tqdm
from ....globals import HEATMAP,COLORS
from matplotlib.colors import to_rgb




def show(device):
  # CREATE CYCLE GRAPH ###########################################################


  # Here we prepare a graph that is a cycle of n nodes. We then compute all pair-
  # wise graph distances because we'll want to learn an embedding that embeds the
  # vertices of the graph on the surface of a torus, such that the distances of
  # the induced discrete metric space of the graph are preserved as well as
  # possible through the positioning of the embeddings on the torus.

  #n = 20

  training_examples = []
  bht_points = {
    # B
    1 : (0,0),2 : (0,1.5),3 : (0,3),4 : (0,4.5),5  : (0,6),6 : (1,0),7  : (1,3),
    8 : (1,6),9 : (2,0),10  : (2,3),11  : (2,6),12  : (3,1),13  : (3,2),
    14  : (3,4),15  : (3,5),
    # H
    16 : (5,0),17 : (5,1.5),18 : (5,3),19 : (5,4.5),20 : (5,6),21 : (6.5,3),
    22 : (8,0),23 : (8,1.5),24 : (8,3),25 : (8,4.5),26 : (8,6),
    # T
    27 : (10,6),28 : (11.5,0),29 : (11.5,1.5),30 : (11.5,3),31 : (11.5,4.5),
    32 : (11.5,6),33 : (13,6)
  }

  n = bht_points.__len__()
  for i in range(n):
    for j in range(i):
      d = torch.norm(torch.tensor(bht_points[i+1],dtype=torch.float32) - torch.tensor(bht_points[j+1],dtype=torch.float32))
      d = d * (2 * pi * 0.3) / 12
      training_examples.append((i,j,d))


  # for i in range(n):
  #   # only consider pair-wise distances below diagonal of distance matrix
  #   for j in range(i):
  #     # determine distance between vertice i and j
  #     d = i-j
  #     if d > n//2:
  #       d = n-d
  #     # scale down distance
  #     d = d * ((2 * pi * 0.3) / (n-1))
  #     # add edge and weight to training examples
  #     training_examples.append((i,j,d))

  # the training_examples now consist of a list of triplets (v1, v2, d)
  # where v1, v2 are vertices, and d is their (scaled) graph distance


  # CREATION OF PRODUCT SPACE (TORUS) ############################################


  # create first sphere manifold of radius 1 (default)
  # (the Exact version uses the exponential map instead of the retraction)
  r1      = 1.0
  sphere1 = SphereExact().to(device)

  # create second sphere manifold of radius 0.3
  r2      = 0.3
  sphere2 = Scaled(SphereExact(), scale=r2).to(device)

  # create torus manifold through product of two 1-dimensional spheres (actually
  # circles) which are each embedded in a 2D ambient space
  torus = ProductManifold((sphere1, 2), (sphere2, 2)).to(device)


  # INITIALIZATION OF EMBEDDINGS #################################################

  # init embeddings. sidenote: this initialization was mostly chosen for
  # illustration purposes. you may want to consider better initialization
  # strategies for the product space that you'll consider.
  X = (torch.randn(n, 4).abs()*0.5).to(device)
  

  # augment embeddings tensor to a manifold tensor with a reference to the product
  # manifold that they belong to such that the optimizer can determine how to
  # convert the the derivatives of pytorch to the correct Riemannian gradients
  X = ManifoldTensor(X, manifold=torus)

  # project the embeddings onto the spheres' surfaces (in-place) according to the
  # orthogonal projection from ambient space onto the sphere's surface for each
  # spherical factor
  X.proj_()

  # declare the embeddings as trainable manifold parameters
  X = ManifoldParameter(X)


  # PLOTTING FUNCTIONALITY #######################################################


  # array storing screenshots
  screenshots = []

  # torus surface
  phi, theta = np.mgrid[0.0:2.0 * pi:100j, 0.0:2.0 * pi:100j]
  torus_x = cos(phi) * (r1 + r2 * cos(theta))
  torus_y = sin(phi) * (r1 + r2 * cos(theta))
  torus_z = r2 * sin(theta)

  # embedding point surface
  ball_size = 0.035
  u       = np.linspace(0, 2 * pi, 100)
  v       = np.linspace(0, pi, 100)
  ball_x  = ball_size * np.outer(cos(u), sin(v))
  ball_y  = ball_size * np.outer(sin(u), sin(v))
  ball_z  = ball_size * np.outer(np.ones(np.size(u)), cos(v))


  def plot_point(x, y, z):
    mlab.mesh(x + ball_x, y + ball_y, z + ball_z, color=to_rgb(COLORS.orange))


  def update_plot(X):
    # transform embedding (2D X 2D)-coordinates to 3D coordinates on torus
    cos_phi = X[:,0] * r1
    sin_phi = X[:,1] * r1
    xx = X[:,0] + cos_phi * X[:,2] * r2
    yy = X[:,1] + sin_phi * X[:,2] * r2
    zz = r2 * X[:,3]

    # create figure
    mlab.figure(size=(700, 500), bgcolor=(1, 1, 1))

    # plot torus surface
    mlab.mesh(torus_x, torus_y, torus_z, color=to_rgb(COLORS.petrol), opacity=0.5)

    # plot embedding points on torus surface
    for i in range(n):
      plot_point(xx[i], yy[i], zz[i])

    # save screenshot
    mlab.view(azimuth=0, elevation=60, distance=4, focalpoint=(0, 0, -0.2))
    mlab.gcf().scene._lift()
    screenshots.append(mlab.screenshot(antialiased=True))
    mlab.close()


  # TRAINING OF EMBEDDINGS IN PRODUCT SPACE ######################################


  # build RADAM optimizer and specify the embeddings as parameters.
  # note that the RADAM can also optimize parameters which are not attached to a
  # manifold. then it just behaves like the usual ADAM for the Euclidean vector
  # space. we stabilize the embedding every 1 steps, which rthogonally projects
  # the embedding points onto the manifold's surface after the gradient-updates to
  # ensure that they lie precisely on the surface of the manifold. this is needed
  # because the parameters may get slightly off the manifold's surface for
  # numerical reasons. Not stabilizing may introduce small errors that are added
  # up over time.
  riemannian_adam = RiemannianAdam(params=[X], lr=1e-2, stabilize=1)

  # we'll just use this as a random examples sampler to get some stochasticity
  # in our gradient descent
  num_training_examples = len(training_examples)
  training_example_indices = np.array(range(num_training_examples))
  def get_subset_of_examples():
    return list(np.random.choice(training_example_indices,
                                 size=int(num_training_examples/4),
                                 replace=True))

  # training loop to optimize the positions of embeddings such that the
  # distances between them become as close as possible to the true graph distances
  for t in tqdm(range(1000)):
    # zero-out the gradients
    riemannian_adam.zero_grad()

    # compute loss for next batch
    loss = torch.tensor(0.0)
    indices_batch = get_subset_of_examples()
    for i in indices_batch:
      v_i, v_j, target_distance = training_examples[i]
      # compute the current distances between the embeddings in the product
      # space (torus)
      current_distance = torus.dist(X[v_i,:], X[v_j,:])
      # add squared loss of current and target distance to the loss
      loss += (current_distance - target_distance).pow(2)

    # compute derivative of loss w.r.t. parameters
    loss.backward()

    # let RADAM compute the gradients and do the gradient step
    riemannian_adam.step()

    # plot current embeddings
    with torch.no_grad():
      update_plot(X.detach().numpy())


  # CREATE ANIMATED GIF ##########################################################
  #return screenshots
  imageio.mimsave(f'training.gif', screenshots, duration=1/24)
