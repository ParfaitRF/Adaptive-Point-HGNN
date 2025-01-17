import numpy as np

R = lambda yaw: np.array([[np.cos(yaw),  0,  np.sin(yaw)],
                      [0,            1,  0          ],
                      [-np.sin(yaw), 0,  np.cos(yaw)]])

# output image size
IMG_HEIGHT  = 375
IMG_WIDTH   = 1242
