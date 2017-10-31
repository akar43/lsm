# Learnt Stereo Machines
This is a Tensorflow implementation of Learnt Stereo Machines as presented in the NIPS 2017 paper below. It supports training, validation and testing of Voxel LSMs and Depth LSMs on the ShapeNet dataset.

[**Learning a Multi-view Stereo Machine**](https://people.eecs.berkeley.edu/~akar/deepmvs.pdf)<br>
[Abhishek Kar](https://people.eecs.berkeley.edu/~akar/), [Christian Häne](https://people.eecs.berkeley.edu/~chaene/), [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/) <br>
NIPS 2017 

![LSM](https://people.eecs.berkeley.edu/~akar/lsm/images/Network.png)

## Setup
### Prerequisites
 - Linux or OSX
 - NVIDIA GPU + CUDA + CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification, but untested)

### Prepare data
The system requires rendered images, depth maps (for D-LSMs), intrinsic/extrinsic camera matrices and voxelizations of the 3D models for training. A version of these renders and cameras can be downloaded using the provided script.
```
bash prepare_data.sh
```

### Setup virtualenv
We recommend using virtualenv to run experiments without modifying your global python distribution.
```
cd <project root>
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

### Optional setup
You might want to specify the GPU to use for experiments (in a multi GPU machine) and suppress TF messages before running scripts. The project root also needs to be added to `PYTHONPATH`.
```
export CUDA_VISIBLE_DEVICES=<GPU to run on> #Specify GPU to run on
export PYTHONPATH=`pwd`:$PYTHONPATH #Add project root to PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=2 #Suppress extra messages from TF
```

## Voxel LSM (V-LSM)
![VLSM](https://people.eecs.berkeley.edu/~akar/lsm/images/voxel_results.png)

### Training
Training a V-LSM on ShapeNet with default arguments. Model checkpoints and tensorboard logs are written out to a unique directory created by default within `./log` displayed at the top after starting training.
```
python voxels/train_vlsm.py --argsjs args/args_vlsm.json
```

### Testing
Testing a Voxel LSM on ShapeNet.
```
LOG=<log directory used while training. e.g. ./log/2017-10-30_132841/train>
CHECKPOINT=<checkpoint to evaluate. e.g. mvnet-100000>

python voxels/test_vlsm.py --log $LOG --ckpt $CHECKPOINT --test_split_file data/splits.json
```
### Validation
You can also choose to run continuous validation while the model is training using the following command. This should add new fields to tensorboard showing validation accuracy/error.
```
LOG=<log directory used while training. e.g. ./log/2017-10-30_132841/train>

python voxels/val_vlsm.py --log $LOG --val_split_file data/splits.json
```

### Viewing progress on Tensorboard
You can view the training progress on tensorboard by using the logs written out while training.
```
LOG=<log directory used while training. e.g. ./log/2017-10-30_132841/train>
tensorboard --logdir $LOG
```


## Depth LSM (D-LSM)
The instructions for D-LSMs are very similar to V-LSMs. You can perform training, validation and testing using the following scripts as well as visualize progress on Tensorboard.

![DLSM](https://people.eecs.berkeley.edu/~akar/lsm/images/depth_results.png)

### Training
```
python depth/train_dlsm.py --argsjs args/args_dlsm.json
```

### Validation
```
LOG=<log directory used while training. e.g. ./log/2017-10-30_132841/train>

python depth/val_dlsm.py --log $LOG --val_split_file data/splits.json
```

### Testing
```
LOG=<log directory used while training. e.g. ./log/2017-10-30_132841/train>
CHECKPOINT=<checkpoint to evaluate. e.g. mvnet-100000>

python depth/test_dlsm.py --log $LOG --ckpt $CHECKPOINT --test_split_file data/splits.json
```
## Citation
```
@incollection{lsmKarHM2017,
  author = {Abhishek Kar and
  Christian H\"ane and
  Jitendra Malik},
  title = {Learning a Multi-View Stereo Machine},
  booktitle = NIPS,
  year = {2017},
  }
```