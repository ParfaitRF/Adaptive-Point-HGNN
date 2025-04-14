# Adaptive Point-HGNN

This repository is complementary to the [thesis thesis](thesis/main.pdf). If you find the code useful, consider citing our work

```plaintext
@inproceedings{prf2025acehgpnn,
    author    	= {Parfait R. Fejou},
    title     	= {Adaptive Curvature Exploration in Non Euclidean Point-GNNs},
    booktitle 	= {Object Detection in Non Euclidean Point-GNN with Adaptive Curvature Exporation},
    year      	= {2025},
    pages     	= {1-60},
    publisher 	= {Berlin University of Applied Sciences}
    url 	= {https://github.com/ParfaitRF/Adaptive-Point-HGNN}
}
```

## Getting Started

### Downloading the repository

```plaintext
git clone https://github.com/ParfaitRF/Adaptive-Point-HGNN.git --recursive
```

### Prerequisites

The required packages can be installed from the requirements file using `pip install -r requirements.txt` also the anaconda environment can be found in the `env` folder.

### KITTI Dataset

```plaintext
DATASET_ROOT_DIR
├── kitti                    	# Left color images
│   ├── 3DOP_splits
|   |   |
|   |   :
|   |
│   └── testing 		# test set
|   |   |
|   |   |── calib		# calibration files
|   |   |── image_2		# image data
|   |   └── velodyne		# point cloud data
|   |
|   └── training		# training set
|       |
|       |── calib
|       |── image_2
|       |── label_2
|       └── velodyne
|
├── kitti_dataset.py         	# Velodyne point cloud files
├── transformations.py		# transformation functionalities
└── utils.py			# utilities
```

The file `kitti_dataset.py` holds the class `KittiDataset` which can be used as an interface to the dataset. We recommend exploring the [notebook](notebook.ipynb)
