""" ase manifold from which hyperbolic, poincare and euclidean are derived."""

from torch.nn import Parameter

class Manifold(object):
  """ Abstract class to define operations carried out on a manifold."""

  # ===========================================================================#
  # ================================== GETTER =================================#
  # ===========================================================================#
  def __init__(self):
    super().__init__()
    self.eps = 10e-8

  def sqdist(self, p1, p2, c):
    """ Computes squared distance between pairs of points.
    @param p1: first point.
    @param p1: second point.
    @param c: TODO.

    @return: squared distance between pairs of points.
    """
    raise NotImplementedError

  def egrad2rgrad(self, p, dp, c):
    """ Converts Euclidean Gradient to Riemannian Gradients.
    @param p: first point.
    @param dp: second point.
    @param c: TODO.

    @return converted gradient
    """
    raise NotImplementedError

  def proj(self, p, c):
    """ Projects point p onto the manifold.
    
    @param p: point to be projected.
    @param c: TODO.

    @return: projected point.
    """
    raise NotImplementedError

  def proj_tan(self, u, p, c):
    """ Projects u on the tangent space of p.
    
    @param u: point to be projected
    @param p: point on which tangent space is computed.
    @param c: TODO.

    @return: projected point.
    """
    raise NotImplementedError

  def proj_tan0(self, u, c):
    """ Projects u on the tangent space of the origin.
    
    @param u: first point.
    @param c: TODO.

    @return: projected point.
    """
    raise NotImplementedError

  def expmap(self, u, p, c):
    """Computes exponential map of u at point p.
    
    @param u: point for which map is to be compued
    @param p: point at which is is to be computed
    @param c: curvature.

    @return: result of map.
    """
    raise NotImplementedError

  def logmap(self, p1, p2, c):
    """Computes logarithmic map of p1 at point p2.
    
    @param p1: point for which map is to be compued
    @param p2: point at which is is to be computed
    @param c: curvature.

    @return: result of map.
    """
    raise NotImplementedError

  def expmap0(self, u, c):
    """Computes exponential map of u at origin.
    
    @param u: point for which map is to be compued
    @param c: curvature.

    @return: result of map.
    """
    raise NotImplementedError

  def logmap0(self, p, c):
    """Computes logarithmic map of u at point p.
    
    @param p: point for which map is to be compued
    @param c: curvature.

    @return: result of map.	
    """
    raise NotImplementedError

  def mobius_add(self, x, y, c, dim=-1):
    """ Computes the möbius addition of x and y.

    @param x: first point.
    @param y: second point.
    @param c: curvature
    @param dim: dimension of the manifold.

    @return: result of addition.
    """
    raise NotImplementedError

  def mobius_matvec(self, m, x, c):
    """Performs hyperboic martrix-vector multiplication.
    
    @param m: matrix to be multiplied.
    @param x: point to be multiplied.
    @param c: curvature.

    @return: result of multiplication.
    """
    raise NotImplementedError
  
  def inner(self, p, c, u, v=None, keepdim=False):
    """ Compues the inner product for tangent vectors at point x.
    
    @param p: point at which inner product is computed.
    @param c: curvature.
    @param u: first vector.
    @param v: second vector.
    @param keepdim: if True, the output will have dim retained.

    @return: computed inner product.
    """
    raise NotImplementedError

  def ptransp(self, x, y, u, c):
    """ Parallel transport of u from x to y.
    
    @param x: point from which u is transported.
    @param y: point to which u is transported.
    @param u: vector to be transported.
    @param c: curvature.

    @return: transported vector.
    """
    raise NotImplementedError

  def ptransp0(self, y, u, c):
    """Parallel transport of u from the origin to y.
    
    @param x: point from which u is transported.
    @param u: vector to be transported.
    @param c: curvature.

    @return: transported vector.
    """
    raise NotImplementedError
  
  # ===========================================================================#
  # ================================== SETTER =================================#
  # ===========================================================================#

  def init_weights(self, w, c, irange=1e-5):
    """Initializes random weigths on the manifold.
    
    @param w: weights to be initialized.
    @param c: curvature.
    @param irange: range of initialization.
    """
    raise NotImplementedError

  
  

class ManifoldParameter(Parameter):
  """
  Subclass of torch.nn.Parameter for Riemannian optimization.
  """
  def __new__(cls, data, requires_grad, manifold, c):
    return Parameter.__new__(cls, data, requires_grad)

  def __init__(self, data, requires_grad, manifold, c):
    self.c = c
    self.manifold = manifold

  def __repr__(self):
    return '{} Parameter containing:\n'.format(self.manifold.name) + super(Parameter, self).__repr__()