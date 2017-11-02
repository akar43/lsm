# Learnt Stereo Machines
This is a Tensorflow implementation of Learnt Stereo Machines as presented in the NIPS 2017 paper below. It supports training, validation and testing of Voxel LSMs and Depth LSMs on the ShapeNet dataset.

**Learning a Multi-view Stereo Machine**<br>
[Abhishek Kar](https://people.eecs.berkeley.edu/~akar/), [Christian Häne](https://people.eecs.berkeley.edu/~chaene/), [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/) <br>
NIPS 2017<br>
[**[blog]**](http://bair.berkeley.edu/blog/2017/09/05/unified-3d/) 
[**[paper]**](https://people.eecs.berkeley.edu/~akar/lsm/lsm_nips17.pdf)
[**[arxiv]**](https://arxiv.org/abs/1708.05375)

![LSM](https://people.eecs.berkeley.edu/~akar/lsm/images/Network.png)

## Setup
### Prerequisites
 - Linux or OSX (Tested on Ubuntu 16.04)
 - NVIDIA GPU + CUDA + CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification, but untested)

### Prepare data
The system requires rendered images, depth maps (for D-LSMs), intrinsic/extrinsic camera matrices and voxelizations of the 3D models for training. A version of these renders and cameras can be downloaded using the provided script. WARNING: The full release is large (21G tar file) and will take some time to unpack. Use the following script to download all the data required to train LSMs from scratch.
```
bash prepare_data.sh
```
If you are interested in only the voxelizations of the models, we also make them available at the links below.

[**[ShapeNet voxels (32^3 and 64^3) (58M)]**](http://people.eecs.berkeley.edu/~akar/lsm/shapenet_voxels.tar.gz)
[**[ShapeNet renderings + voxels (21G)]**](http://people.eecs.berkeley.edu/~akar/lsm/shapenet_release.tar.gz)

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

## Pretrained Models
We are releasing pretrained models for V-LSMs and D-LSMs trained on the ShapeNet dataset which can be used to reproduce numbers from the paper. Note that the numbers might differ a little (higher for the code release) due to minor changes in the code after submission. The models can be downloaded with the tensorboard run logs (1.7G) or without (70M) and can be downloaded from the links below. You can also use the [`get_models.sh`](https://github.com/akar43/lsm/blob/master/get_models.sh) script to download the models.

[**[LSM v1 (with logs)]**](https://people.eecs.berkeley.edu/~akar/lsm/models_lsm_v1.tar.gz) 
[**[LSM v1 (models only)]**](https://people.eecs.berkeley.edu/~akar/lsm/models_lsm_v1_small.tar.gz)

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

## Viewing progress on Tensorboard
<div width=100%>
<img src="https://people.eecs.berkeley.edu/~akar/lsm/images/run_vlsm.png" width=49%>
<img src="https://people.eecs.berkeley.edu/~akar/lsm/images/run_vlsm2.png" width=49%>
<img src="https://people.eecs.berkeley.edu/~akar/lsm/images/run_dlsm.png" width=98%>
</div>

You can view the training progress on tensorboard by using the logs written out while training.
```
LOG=<log directory used while training. e.g. ./log/2017-10-30_132841/train>
tensorboard --logdir $LOG
```



## Citation
If you use our code, we request you to cite the following work.
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
<div align=center>
<img src="https://people.eecs.berkeley.edu/~akar/lsm/images/proj_gif.gif" width=45% style="margin-left:4%; border-right:solid; border-width:1px; border-color:rgba(0,0,0,0.42);">
<img src="https://people.eecs.berkeley.edu/~akar/lsm/images/unproj_gif.gif" width=45% style="margin-right:4%">
</div>