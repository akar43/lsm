import argparse
import logging
import os.path as osp

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import SHAPENET_IM
from loader import pad_batch
from models import grid_nets, im_nets, model_dlsm
from mvnet import MVNet
from ops import conv_rnns
from skimage.io import imsave
from shapenet import ShapeNet
from utils import init_logging, process_args, get_session_config, mkdir_p


def vis_depth(pred, gt, sid, mid, view_idx):
    for ix in range(gt.shape[0]):
        sd = osp.join(args.savedir, sid[ix, 0], mid[ix, 0])
        mkdir_p(sd)
        for jx in range(gt.shape[1]):
            # save_d = np.hstack([pred[ix, 0, jx] / 10.0, gt[ix, jx] / 10.0])
            save_d = pred[ix, 0, jx] / 10.0
            save_d = np.clip(save_d, 0.0, 1.0)
            imsave(
                osp.join(sd, 'pred_{:d}.png'.format(view_idx[ix, jx])),
                save_d[..., 0])


def run(args):
    net = MVNet(
        vmin=-0.5,
        vmax=0.5,
        vox_bs=args.test_batch_size,
        im_bs=args.test_im_batch,
        grid_size=args.nvox,
        im_h=args.im_h,
        im_w=args.im_w,
        mode="TEST",
        norm=args.norm)

    im_dir = SHAPENET_IM
    # Setup network
    net = model_dlsm(
        net,
        im_nets[args.im_net],
        grid_nets[args.grid_net],
        conv_rnns[args.rnn],
        im_skip=args.im_skip,
        ray_samples=args.ray_samples,
        sepup=args.sepup,
        proj_x=args.proj_x,
        proj_last=True)

    net.print_net()
    sess = tf.Session(config=get_session_config())
    saver = tf.train.Saver()
    saver.restore(sess, osp.join(args.log, args.ckpt))
    coord = tf.train.Coordinator()

    # Init dataset
    dset = ShapeNet(im_dir=im_dir, split_file=args.test_split_file, rng_seed=1)
    mids = dset.get_smids(args.split)
    print('Testing {:d} models'.format(len(mids)))
    items = ['shape_id', 'model_id', 'im', 'K', 'R', 'depth', 'view_idx']
    dset.init_queue(
        mids,
        args.test_im_batch,
        items,
        coord,
        nepochs=1,
        qsize=32,
        nthreads=args.prefetch_threads)

    # Testing loop
    pbar = tqdm(desc='Testing', total=len(mids))
    deq_mids, deq_sids, deq_view_idx = [], [], []
    try:
        while not coord.should_stop():
            batch_data = dset.next_batch(items, net.batch_size)
            if batch_data is None:
                continue
            deq_sids.append(batch_data['shape_id'])
            deq_mids.append(batch_data['model_id'])
            deq_view_idx.append(batch_data['view_idx'])
            num_batch_items = batch_data['K'].shape[0]
            batch_data = pad_batch(batch_data, args.test_batch_size)
            feed_dict = {
                net.K: batch_data['K'],
                net.Rcam: batch_data['R'],
                net.ims: batch_data['im']
            }

            pred = sess.run(net.depth_out, feed_dict=feed_dict)
            vis_depth(pred, batch_data['depth'], batch_data['shape_id'],
                      batch_data['model_id'], batch_data['view_idx'])
            # Update iou dict
            pbar.update(num_batch_items)
    except Exception, e:
        logger.error(repr(e))
        dset.close_queue(e)
    finally:
        pbar.close()
        sess.close()
        logger.info('Testing completed')
        coord.join()

    deq_mids = np.concatenate(deq_mids, axis=0)[..., 0]
    deq_sids = np.concatenate(deq_sids, axis=0)[..., 0]
    deq_view_idx = np.concatenate(deq_view_idx, axis=0)
    sort_idx = np.argsort(deq_mids)
    deq_mids = deq_mids[sort_idx]
    deq_sids = deq_sids[sort_idx]
    deq_view_idx = deq_view_idx[sort_idx, :]
    with open(args.test_set_file, 'w') as f:
        for ix in range(len(deq_mids)):
            f.write(deq_sids[ix] + '\t' + deq_mids[ix] + '\t' +
                    ' '.join(map(str, deq_view_idx[ix].tolist())) + '\n')
    print('Test set file: {:s}'.format(args.test_set_file))


def parse_args():
    parser = argparse.ArgumentParser(description='Options for MVNet')
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--test_im_batch', type=int, default=4)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--test_set_file', type=str, default=None)
    parser.add_argument('--loglevel', type=str, default='info')
    parser.add_argument(
        '--test_split_file', type=str, default='data/splits.json')
    parser.add_argument('--prefetch_threads', type=int, default=2)
    parser.add_argument('--savedir', type=str, default=None)
    args = process_args(parser)
    return args


if __name__ == '__main__':
    args = parse_args()
    init_logging(args.loglevel)
    logger = logging.getLogger('mview3d.' + __name__)
    run(args)
