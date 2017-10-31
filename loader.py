import numpy as np
import skimage
from scipy.io import loadmat
from skimage.io import imread
from skimage.transform import resize


def read_im(f):
    im = skimage.img_as_float(imread(f))
    if im.ndim == 2:
        im = np.stack([im, im, im], axis=2)

    if im.shape[-1] == 4:
        alpha = np.expand_dims(im[..., 3], 2)
        im = im[..., :3] * alpha + (1 - alpha)
    return im[..., :3]


def read_depth(f):
    im = skimage.img_as_float(imread(f)) * 10
    if im.ndim == 2:
        im = np.expand_dims(im, 2)
    return im


def read_camera(f):
    cam = loadmat(f)
    Rt = cam['extrinsic'][:3]
    K = cam['K']
    return K, Rt


def read_quat(f):
    cam = loadmat(f)
    q = cam['quat'].ravel()
    return q


def read_vol(f, tsdf=False):
    def get_vox(f):
        try:
            data = loadmat(f, squeeze_me=True)
        except:
            print('Error reading {:s}'.format(f))
            return None
        vol = np.transpose(data['Volume'].astype(np.bool), [0, 2, 1])
        vol = vol[:, ::-1, :]
        return vol

    def get_tsdf(f, trunc=0.2):
        try:
            data = loadmat(f, squeeze_me=True)
        except:
            print('Error reading {:s}'.format(f))
            return None

        tsdf = data['tsdf']
        tsdf[tsdf < -trunc] = -trunc
        tsdf[tsdf > trunc] = trunc
        tsdf = np.transpose(tsdf, [0, 2, 1])
        tsdf = tsdf[:, ::-1, :]
        return tsdf

    load_func = get_tsdf if tsdf else get_vox
    vol = load_func(f).astype(np.float32)
    vol = vol[..., np.newaxis]
    return vol


def pad_batch(batch_data, bs):
    for k, v in batch_data.iteritems():
        n = v.shape[0]
        to_pad = bs - n
        if to_pad == 0:
            continue
        pad = np.stack([v[0, ...]] * to_pad, axis=0)
        batch_data[k] = np.concatenate([v, pad], axis=0)

    return batch_data


def subsample_grid(grids, sub_ratio):
    def subsample(g):
        ss = np.array(g.shape) / sub_ratio
        sub_grid = np.zeros(ss, dtype=np.bool)
        for ix in range(sub_ratio):
            for jx in range(sub_ratio):
                for kx in range(sub_ratio):
                    sub_grid = np.logical_or(
                        sub_grid,
                        g[ix::sub_ratio, jx::sub_ratio, kx::sub_ratio])
        return sub_grid

    return [subsample(g) for g in grids]