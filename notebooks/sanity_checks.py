from notebook_res import *

dim,n_samples = 2,5
x_sample  = np.random.random((n_samples,dim))
y_sample  = np.random.random((n_samples,dim))
r_sample  = np.random.random(n_samples)

def check_mobius_addition():
  res_plus  = mobius_add(x_sample,y_sample,0)
  _sum      = x_sample + y_sample
  assert np.allclose(res_plus,_sum), f"mobius addition check failed!"

def check_scale():
  res_scale = mobius_scale(r_sample,x_sample[0],0)
  assert np.allclose(res_scale,r_sample*x_sample), f"mobius_scale check failed!"

def check_mobius_distance():
  res_dist  = distance(y_sample,x_sample,0)
  diff = x_sample-y_sample
  diff_norm = norm(diff,axis=1)
  assert np.allclose(res_dist,2*diff_norm), f"mobius distance check failed!"
