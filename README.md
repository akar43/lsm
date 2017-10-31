# Learnt Stereo Machines

## Data

## Virtualenv setup

```
cd <project root>
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

## Voxel LSM (V-LSM)

Training a Voxel LSM on ShapeNet with default args
```
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=`pwd`:$PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=2

python voxels/train_vlsm.py --argsjs args/args_vlsm.json
```

Testing a Voxel LSM on ShapeNet with batch size 4 and 4 views per model
```
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=`pwd`:$PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=2

LOG=<log directory used while training. e.g. ./log/2017-10-30_132841/train>
CHECKPOINT=<checkpoint to evaluate. e.g. mvnet-100000>

python voxels/test_vlsm.py --log $LOG --test_batch_size 4 --test_im_batch 4 --eval_thresh 0.4 --ckpt $CHECKPOINT --split test --test_split_file data/splits.json
```

Running validation while training a model
```
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=`pwd`:$PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=2

LOG=<log directory used while training. e.g. ./log/2017-10-30_132841/train>

python voxels/val_vlsm.py --log $LOG --val_batch_size 4 --val_im_batch 4 --eval_thresh 0.4 --val_split_file data/splits.json
```

Viewing progress on tensorboard

```
LOG=<log directory used while training. e.g. ./log/2017-10-30_132841/train>
tensorboard --logdir $LOG
```


## Depth LSM (D-LSM)