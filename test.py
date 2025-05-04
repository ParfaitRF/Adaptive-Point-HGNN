import os
import random
from matplotlib import pyplot as plt

from data.kitti_dataset import KittiDataset
from notebooks.sanity_checks import test_decoder
from models.box_encoders_decoders import (
  classaware_all_class_box_encoding,
  classaware_all_class_box_decoding,
  classaware_all_class_box_canonical_encoding,
  classaware_all_class_box_canonical_decoding
)

kitti = KittiDataset() # initialize data object

encoder_fn = classaware_all_class_box_encoding
decoder_fn = classaware_all_class_box_decoding

test_decoder(encoder_fn, decoder_fn)



# kitti.vis_crop_aug_sampler()