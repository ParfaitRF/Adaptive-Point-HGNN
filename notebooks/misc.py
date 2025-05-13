import json
import numpy as np
from models.graph_gen import (
  multi_layer_downsampling,
  gen_disjointed_rnn_local_graph_v3,
  gen_multi_level_local_graph_v3
)

import open3d as o3d
from open3d.visualization import draw

dtype_f = o3d.core.float32
dtype_i = o3d.core.int32
device  = o3d.core.Device("CPU:0")


